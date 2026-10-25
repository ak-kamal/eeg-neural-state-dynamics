"""Pooled TDE-HMM for latent neural-state discovery.

Implements the pipeline described in Vidaurre et al. (2016, NeuroImage) and
Vidaurre et al. (2018, Nature Communications): time-delay embedding (TDE)
of the multi-channel EEG, PCA dimensionality reduction, and a Gaussian
Hidden Markov Model trained on data pooled across subjects.

The model is fitted on concatenated trials from all subjects in the
development subset, so the same state dictionary is used for every subject.
This is the "group-level" or "pooled" approach used by the OHBA TDE-HMM
pipeline (osl-dynamics) and the HMM-MAR toolbox.

Key references:
    Vidaurre, D. et al. (2016). Spectrally resolved fast transient brain
        states in functional magnetic resonance data. NeuroImage, 128, 395-407.
    Vidaurre, D. et al. (2018). Spontaneous cortical activity transiently
        organises into frequency specific phase-coupling networks.
        Nature Communications, 9, 2987.
    Quinn, A. J. et al. (2018). Task-evoked dynamic network analysis through
        hidden Markov modeling. Frontiers in Neuroscience, 12, 603.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.decomposition import PCA

from . import config

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each feature to zero mean and unit variance.

    Full-covariance Gaussian HMMs are numerically unstable when the input
    features have very small variance, as is the case for PCA time courses
    of volt-scale EEG (variance ~1e-8). Standardization is the standard
    pre-step in TDE-HMM pipelines (Vidaurre et al. 2016; the OHBA
    osl-dynamics pipeline normalizes PCA outputs before fitting).

    Returns
    -------
    X_std : standardized data
    mean  : per-feature means (needed to invert the transform)
    std   : per-feature standard deviations
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std

# --- Time-delay embedding --------------------------------------------------

def time_delay_embed(
    data: np.ndarray,
    lags: list[int] | None = None,
) -> np.ndarray:
    """Construct a time-delay embedded representation of the data.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Input time series (one trial or one continuous segment).
    lags : list of int, optional
        Sample lags to include. Defaults to ``config.TDE_LAGS``
        (currently ±7 samples ≈ 56 ms window at 250 Hz).

    Returns
    -------
    embedded : np.ndarray, shape (n_channels * n_lags, n_times_valid)
        The TDE representation. Time points where the full lag window is
        not available (the first ``|min(lag)|`` and last ``max(lag)``
        samples) are trimmed.

    Notes
    -----
    The TDE representation at time ``t`` is the concatenation of the signal
    at ``t + lag`` for every lag in ``lags``, for every channel. This gives
    the model access to the local spectral content of the signal, because
    the autocovariance structure of the embedded vector encodes
    oscillatory power at frequencies corresponding to the lag spacing
    (Vidaurre et al., 2016).
    """
    lags = lags if lags is not None else config.TDE_LAGS
    n_channels, n_times = data.shape

    # Keep only time points where all lags are valid.
    valid_start = max(0, -min(lags))
    valid_end = n_times - max(0, max(lags))
    if valid_end <= valid_start:
        raise ValueError(
            f"Signal too short ({n_times} samples) for lag range "
            f"[{min(lags)}, {max(lags)}]."
        )

    n_lags = len(lags)
    n_valid = valid_end - valid_start
    embedded = np.zeros((n_channels * n_lags, n_valid), dtype=data.dtype)

    for k, lag in enumerate(lags):
        chunk = data[:, valid_start + lag : valid_end + lag]
        embedded[k * n_channels : (k + 1) * n_channels, :] = chunk

    return embedded


# --- PCA reduction ---------------------------------------------------------

def fit_pca(
    embedded: np.ndarray,
    n_components: int | None = None,
    random_state: int | None = None,
) -> PCA:
    """Fit PCA on the TDE representation.

    PCA is applied after TDE to reduce the ~900-dimensional embedded vector
    to a tractable dimensionality for the Gaussian emission model. The
    number of components is a configurable parameter (``config.N_PCS``).

    Parameters
    ----------
    embedded : np.ndarray, shape (n_features, n_samples)
        TDE matrix, channels × lags by time.
    n_components : int, optional
        Number of principal components to retain. Defaults to
        ``config.N_PCS``.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    """
    n_components = n_components if n_components is not None else config.N_PCS
    rs = random_state if random_state is not None else config.RANDOM_SEED

    # PCA expects samples × features.
    X = embedded.T
    pca = PCA(n_components=n_components, random_state=rs)
    pca.fit(X)
    return pca


def transform_pca(embedded: np.ndarray, pca: PCA) -> np.ndarray:
    """Apply a fitted PCA to a TDE matrix.

    Returns
    -------
    reduced : np.ndarray, shape (n_samples, n_components)
        PCA-reduced data, ready for HMM fitting.
    """
    return pca.transform(embedded.T)


# --- Pooled data preparation ----------------------------------------------

def prepare_pooled_data(
    subject_trials: dict[str, list[np.ndarray]],
    pca: PCA | None = None,
) -> tuple[np.ndarray, list[int], PCA]:
    """TDE → PCA (streaming) → per-subject z-score → pooled array.

    Uses IncrementalPCA fed trial-by-trial to avoid materializing the
    full (n_features, total_samples) matrix. At TDE ±25 samples, the
    full matrix would be ~12 GB for the final subset — beyond typical
    laptop RAM. IncrementalPCA produces components numerically
    indistinguishable from batch PCA on this data (same inputs, same
    target dimensionality, batch_size ≫ n_components).

    Two passes over the trials: pass 1 fits PCA, pass 2 transforms and
    standardizes per subject. TDE is recomputed between passes so peak
    memory is bounded by one trial (~30 MB).
    """
    from sklearn.decomposition import IncrementalPCA

    if pca is None:
        pca = IncrementalPCA(
            n_components=config.N_PCS,
            batch_size=2000,
        )
        # Pass 1: fit on embedded trials, one at a time
        for sub, trials in subject_trials.items():
            for t in trials:
                emb = time_delay_embed(t)
                if emb.shape[1] >= config.N_PCS:
                    pca.partial_fit(emb.T)

    # Pass 2: transform, then per-subject z-score
    X_list, lengths = [], []
    for sub, trials in subject_trials.items():
        sub_pca_list = []
        for t in trials:
            emb = time_delay_embed(t)
            sub_pca_list.append(pca.transform(emb.T))
            lengths.append(emb.shape[1])
        sub_pca = np.concatenate(sub_pca_list, axis=0)
        m = sub_pca.mean(axis=0, keepdims=True)
        s = sub_pca.std(axis=0, keepdims=True)
        s[s == 0] = 1.0
        X_list.append((sub_pca - m) / s)

    return np.concatenate(X_list, axis=0), lengths, pca


# --- HMM fitting -----------------------------------------------------------

def fit_hmm(
    X: np.ndarray,
    lengths: list[int],
    n_states: int | None = None,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int | None = None,
) -> hmm.GaussianHMM:
    """Fit a Gaussian HMM on pooled, PCA-reduced TDE data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        PCA-reduced data.
    lengths : list of int
        Trial lengths, so the HMM respects trial boundaries.
    n_states : int, optional
        Number of hidden states (K). Defaults to ``config.K_STATES``.
    covariance_type : str
        ``"full"`` (per-state full covariance) is the standard for
        TDE-HMM (Vidaurre et al., 2016). Options: "full", "diag", "tied",
        "spherical".
    n_iter : int
        Maximum Baum-Welch iterations.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    model : hmmlearn.hmm.GaussianHMM
        Fitted HMM. The attributes ``means_``, ``covars_``,
        ``transmat_``, and ``startprob_`` contain the state statistics.
    """
    n_states = n_states if n_states is not None else config.K_STATES
    rs = random_state if random_state is not None else config.RANDOM_SEED

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=rs,
        init_params="stmc",
        params="stmc",
        min_covar=1e-3,
    )
    model.fit(X, lengths=lengths)
    return model

def select_k(
    results_df: "pd.DataFrame",
    k_ceiling: int = 6,
) -> tuple[int, bool]:
    """Pick K from a K-sweep table using knee detection.

    BIC and AIC are uninformative here — both decrease monotonically across
    the grid, so neither will select by argmin. We fall back to knee
    detection on the log-likelihood curve (Kneedle-style: the point of
    maximum distance from the diagonal connecting the first and last
    fitted K in normalised coordinates). This is a documented, reproducible
    rule and behaves correctly under monotonic BIC.

    Returns (K_chosen, was_monotonic), where was_monotonic flags that the
    information criterion was uninformative and the knee rule did the work.
    """
    import numpy as np

    ks = results_df["K"].to_numpy()
    ll = results_df["log_lik"].to_numpy()
    if len(ks) < 3:
        raise ValueError("Knee detection requires at least 3 fitted K values.")

    order = np.argsort(ks)
    ks = ks[order].astype(float)
    ll = ll[order].astype(float)

    was_monotonic = bool(np.all(np.diff(ll) > 0))
    if not was_monotonic:
        # Log-likelihood not monotone; the raw BIC/AIC minimum is meaningful.
        return int(ks[np.argmin(results_df["BIC"].to_numpy()[order])]), False

    ks_n = (ks - ks.min()) / (ks.max() - ks.min())
    ll_n = (ll - ll.min()) / (ll.max() - ll.min())
    distance_from_diagonal = np.abs(ks_n - ll_n)
    k_chosen = int(ks[np.argmax(distance_from_diagonal)])
    return min(k_chosen, k_ceiling), True

def get_state_time_course(
    model: hmm.GaussianHMM,
    X: np.ndarray,
    lengths: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the state sequence for a given dataset.

    Parameters
    ----------
    model : hmmlearn.hmm.GaussianHMM
        Fitted HMM.
    X : np.ndarray, shape (n_samples, n_features)
        Data to decode.
    lengths : list of int, optional
        Trial lengths. If provided, the posterior probabilities and
        Viterbi path are computed separately for each trial, which
        prevents state carry-over across trial boundaries.

    Returns
    -------
    viterbi : np.ndarray, shape (n_samples,)
        Most likely state at each time point.
    posteriors : np.ndarray, shape (n_samples, n_states)
        Posterior probability of each state at each time point.
    """
    if lengths is not None:
        viterbi_list = []
        posterior_list = []
        start = 0
        for L in lengths:
            segment = X[start : start + L]
            # hmmlearn.decode returns (log_prob, state_sequence).
            _, v = model.decode(segment, algorithm="viterbi")
            _, post = model.score_samples(segment)
            viterbi_list.append(v)
            posterior_list.append(post)
            start += L
        viterbi = np.concatenate(viterbi_list)
        posteriors = np.concatenate(posterior_list, axis=0)
    else:
        _, viterbi = model.decode(X, algorithm="viterbi")
        _, posteriors = model.score_samples(X)

    return viterbi, posteriors