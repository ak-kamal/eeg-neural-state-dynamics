"""Regression tests for src/eeg_state/io.py, pinned to sub-04.

These assertions capture the exact counts produced by the dataset audit.
If a future refactor changes the event deduplication or trial
reconstruction logic, these tests will catch it immediately.
"""

import numpy as np
import pandas as pd
import pytest

from eeg_state.config import DATA_RAW
from eeg_state.io import (
    ENCODING_DURATION,
    RETRIEVAL_ONSET_OFFSET,
    load_events,
    reconstruct_trials,
)

SUB = "sub-04"


def test_events_deduplicated():
    ev = load_events(SUB)
    # No (onset, value) pair appears more than once.
    assert not ev.duplicated(subset=["onset", "value"]).any()
    # New Segment entries are dropped.
    assert 99999 not in ev["value"].values


def test_trial_count_is_197():
    # sub-04 has 197 trial onsets (37 at load 15, 40 at every other load).
    # The dataset README states 200 nominal trials; the actual count is
    # what governs downstream analysis.
    trials = reconstruct_trials(load_events(SUB))
    assert len(trials) == 197
    counts = trials["load"].value_counts().sort_index()
    assert counts.to_dict() == {3: 40, 6: 40, 9: 40, 12: 40, 15: 37}


def test_all_trials_responded():
    trials = reconstruct_trials(load_events(SUB))
    assert trials["rt_s"].notna().sum() == 197


def test_accuracy_matches_participants_tsv():
    """Event-derived accuracy must match the published per-load accuracy."""
    trials = reconstruct_trials(load_events(SUB))
    participants = pd.read_csv(DATA_RAW / "participants.tsv", sep="\t")
    row = participants[participants["participant_id"] == SUB].iloc[0]
    acc_by_load = trials.groupby("load")["correct"].mean()
    for load in (3, 6, 9, 12, 15):
        np.testing.assert_allclose(acc_by_load[load], row[f"accuracy_{load}"],
                                   atol=1e-9)


def test_phase_offsets_are_exact():
    trials = reconstruct_trials(load_events(SUB))
    np.testing.assert_allclose(
        trials["retention_onset_s"] - trials["encoding_onset_s"],
        ENCODING_DURATION,
    )
    np.testing.assert_allclose(
        trials["retrieval_onset_s"] - trials["encoding_onset_s"],
        RETRIEVAL_ONSET_OFFSET,
    )