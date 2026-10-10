"""BIDS loading: raw EEG, event deduplication, trial reconstruction.

The events TSV in this dataset lists every event twice (durations 0.0 and
0.001 s at identical onsets). We deduplicate on (onset, value). Triggers
correspond only to trial onset and to the response; retention and retrieval
onsets are reconstructed from the fixed experimental timing documented in
the dataset README.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from typing import Literal

from .config import DATA_RAW, SESSION, TASK

# Fixed experimental timings (dataset README, seconds).
ENCODING_DURATION = 1.5
RETENTION_DURATION = 2.0
RETRIEVAL_DURATION = 1.5
RETRIEVAL_ONSET_OFFSET = ENCODING_DURATION + RETENTION_DURATION  # 3.5 s

TRIAL_CODES = np.array([31, 32, 61, 62, 91, 92, 121, 122, 151, 152])
RESPONSE_CODES = np.array([10, 11])
NEW_SEGMENT_CODE = 99999


# --- Paths -----------------------------------------------------------------

def subject_dir(subject: str, session: str = SESSION) -> Path:
    return DATA_RAW / subject / session / "eeg"


def _bids_prefix(subject: str, session: str = SESSION) -> str:
    return f"{subject}_{session}_task-{TASK}"


# --- Montage ---------------------------------------------------------------

def load_montage(subject: str, session: str = SESSION) -> mne.channels.DigMontage:
    """Build a DigMontage from the BIDS CapTrak electrodes/coordsystem pair."""
    base = subject_dir(subject, session)
    elec_path = base / f"{subject}_{session}_space-CapTrak_electrodes.tsv"
    coord_path = base / f"{subject}_{session}_space-CapTrak_coordsystem.json"

    elec = pd.read_csv(elec_path, sep="\t", encoding="utf-8-sig")
    with open(coord_path, encoding="utf-8-sig") as f:
        coord = json.load(f)

    ch_pos = {
        row["name"]: np.array([row["x"], row["y"], row["z"]], dtype=float)
        for _, row in elec.iterrows()
    }
    lm = coord["AnatomicalLandmarkCoordinates"]
    return mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=np.array(lm["NAS"], dtype=float),
        lpa=np.array(lm["LPA"], dtype=float),
        rpa=np.array(lm["RPA"], dtype=float),
        coord_frame="head",
    )


# --- Raw loading -----------------------------------------------------------

def load_raw(subject: str, session: str = SESSION, preload: bool = True) -> mne.io.BaseRaw:
    """Load a BrainVision raw file and attach the CapTrak montage.

    The original reference (FCz) is not among the recorded channels, so the
    raw data are still in FCz-referenced form. Re-referencing is deliberately
    handled in preprocessing so that bad-channel exclusion can be applied
    first.
    """
    vhdr = subject_dir(subject, session) / f"{_bids_prefix(subject, session)}_eeg.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=preload, verbose="ERROR")
    raw.set_montage(load_montage(subject, session), on_missing="raise")
    return raw


# --- Events ----------------------------------------------------------------

def load_events(subject: str, session: str = SESSION) -> pd.DataFrame:
    """Load and deduplicate the BIDS events TSV.

    Each event appears twice in the raw file (durations 0.0 and 0.001 s at
    identical onsets). We keep the first occurrence of each (onset, value)
    pair. New Segment entries are dropped.
    """
    path = subject_dir(subject, session) / f"{_bids_prefix(subject, session)}_events.tsv"
    ev = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    ev = ev[ev["value"] != NEW_SEGMENT_CODE]
    ev = ev.drop_duplicates(subset=["onset", "value"], keep="first")
    return ev.reset_index(drop=True)


def reconstruct_trials(
    events: pd.DataFrame,
    response_window_s: float = 20.0,
) -> pd.DataFrame:
    """Build a trial-level table with load, probe, accuracy, and reaction time.

    For each trial-onset trigger, the matching response is the first
    response trigger (10 or 11) whose onset lies between retrieval onset
    (trial_onset + 3.5 s) and the earlier of:
      - the next trial onset, and
      - retrieval onset + response_window_s.

    The README states that response time is unlimited, so 20 s is a
    generous safety bound; the effective upper bound is normally the next
    trial onset, because the trial concludes immediately after a response.
    Trials with no response are retained with `correct=None` and `rt_s=NaN`.
    """
    ev = events[events["value"] != NEW_SEGMENT_CODE].reset_index(drop=True)

    trial_mask = ev["value"].isin(TRIAL_CODES).values
    resp_mask = ev["value"].isin(RESPONSE_CODES).values

    t_on = ev.loc[trial_mask, "onset"].to_numpy()
    t_val = ev.loc[trial_mask, "value"].to_numpy()
    r_on = ev.loc[resp_mask, "onset"].to_numpy()
    r_val = ev.loc[resp_mask, "value"].to_numpy()

    rows = []
    for i, (onset, value) in enumerate(zip(t_on, t_val)):
        retrieval = onset + RETRIEVAL_ONSET_OFFSET
        next_trial = t_on[i + 1] if i + 1 < len(t_on) else np.inf
        window_end = min(next_trial, retrieval + response_window_s)

        mask = (r_on >= retrieval) & (r_on < window_end)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            rt, correct = np.nan, None
        else:
            j = idx[0]
            rt = float(r_on[j] - retrieval)
            correct = bool(r_val[j] == 11)

        rows.append({
            "trial_idx": i,
            "onset_s": float(onset),
            "load": int(value // 10),
            "probe_present": bool(value % 10 == 2),
            "correct": correct,
            "rt_s": rt,
            "encoding_onset_s": float(onset),
            "retention_onset_s": float(onset + ENCODING_DURATION),
            "retrieval_onset_s": float(onset + RETRIEVAL_ONSET_OFFSET),
        })

    return pd.DataFrame(rows)

Phase = Literal["encoding", "retention", "retrieval", "outside"]

def classify_phase(t_since_trial_onset: float) -> Phase:
    """Phase of a sample given seconds since trial onset.

    Boundaries follow the dataset README:
        [0.0, 1.5)  encoding
        [1.5, 3.5)  retention
        [3.5, 5.0)  retrieval
        [5.0, inf)  outside (response window / ITI)
    """
    if t_since_trial_onset < 0.0:
        raise ValueError("t_since_trial_onset must be non-negative")
    if t_since_trial_onset < ENCODING_DURATION:
        return "encoding"
    if t_since_trial_onset < ENCODING_DURATION + RETENTION_DURATION:
        return "retention"
    if t_since_trial_onset < RETRIEVAL_ONSET_OFFSET + RETRIEVAL_DURATION:
        return "retrieval"
    return "outside"