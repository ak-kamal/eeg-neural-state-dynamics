"""Batch-preprocess subjects in config.DEV_SUBJECTS.

Produces one cleaned derivative per subject plus a cohort-level QC summary
in results/metrics/batch_preprocessing_qc.csv.

The QC columns follow the conventions used in established batch pipelines:
HAPPE+ER writes per-file data-quality summaries to a single spreadsheet
[Gabard-Durnam et al. 2018]; the MNE-BIDS-Pipeline distinguishes "assess
data quality" from "find bad channels" as separate, logged steps [MNE-BIDS-
Pipeline docs]; CLEAN-EEG emphasizes tracking rank at every stage that can
reduce it [Meyer et al. 2026, NeuroImage].

Failures on individual subjects are caught and logged rather than aborting
the whole batch, so a single corrupt file does not force re-running the
cohort.

Usage:
    uv run python scripts/batch_preprocess.py
"""

import csv
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mne
import numpy as np

from eeg_state import config
from eeg_state import preprocessing as pp
from eeg_state.io import load_raw, load_events, reconstruct_trials

OUT_CSV = config.RESULTS / "metrics" / "batch_preprocessing_qc.csv"


QC_FIELDS = [
    "subject",
    "status",
    "error",
    "n_channels",
    "n_bad_channels",
    "bad_channel_names",
    "rank_after_reference",
    "n_ica_components",
    "n_excluded_components",
    "n_trials",
    "n_trials_with_response",
    "accuracy_overall",
    "rt_median_s",
    "duration_s",
    "sfreq_hz",
    "p95_amplitude_uv",
]


def _nan():
    return float("nan")


def preprocess_one(subject: str) -> dict:
    """Run the pipeline for one subject and return a QC row.

    Any exception is caught by the caller; this function assumes inputs are
    valid.
    """
    raw = pp.highpass(load_raw(subject, preload=True))

    # Rank after referencing is the diagnostic the earlier full-rank ICA bug
    # taught us to track: after average referencing it should equal
    # n_channels - 1, and the ICA must not truncate below that.
    bad_channels = pp.detect_bad_channels(raw)
    raw.info["bads"] = list(bad_channels)
    raw_ref = pp.set_average_reference(raw)
    rank_after_ref = int(mne.compute_rank(raw_ref, rank=None)["eeg"])

    ica_copy = pp.prepare_ica_copy(raw_ref, bad_channels)
    ica = pp.fit_ica(ica_copy)
    labels, exclude = pp.classify_ica_components(ica, ica_copy)
    ica.exclude = list(exclude)

    raw_clean = pp.apply_ica(raw_ref, ica)
    raw_clean = pp.lowpass(raw_clean)
    raw_clean = pp.interpolate_bad_channels(raw_clean)
    raw_clean = pp.downsample(raw_clean)

    out_dir = config.DATA_DERIV / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_preprocessed_raw.fif"
    raw_clean.save(out_path, overwrite=True, verbose="ERROR")

    trials = reconstruct_trials(load_events(subject))
    n_trials = len(trials)
    n_with_response = int(trials["rt_s"].notna().sum())
    accuracy = float(trials["correct"].mean()) if n_with_response else _nan()
    rt_median = float(trials["rt_s"].median()) if n_with_response else _nan()

    amp_uv = np.abs(raw_clean.get_data() * 1e6)

    return {
        "subject": subject,
        "status": "ok",
        "error": "",
        "n_channels": len(raw_clean.ch_names),
        "n_bad_channels": len(bad_channels),
        "bad_channel_names": "|".join(bad_channels),
        "rank_after_reference": rank_after_ref,
        "n_ica_components": int(ica.n_components_),
        "n_excluded_components": len(exclude),
        "n_trials": n_trials,
        "n_trials_with_response": n_with_response,
        "accuracy_overall": round(accuracy, 4),
        "rt_median_s": round(rt_median, 3),
        "duration_s": round(float(raw_clean.times[-1]), 2),
        "sfreq_hz": float(raw_clean.info["sfreq"]),
        "p95_amplitude_uv": round(float(np.percentile(amp_uv, 95)), 3),
    }


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for subject in config.DEV_SUBJECTS:
        print(f"=== {subject} ===")
        try:
            row = preprocess_one(subject)
            print(
                f"  ok | bad_ch={row['n_bad_channels']} | "
                f"ica={row['n_ica_components']} excl={row['n_excluded_components']} | "
                f"p95={row['p95_amplitude_uv']} uV"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            row = {field: "" for field in QC_FIELDS}
            row["subject"] = subject
            row["status"] = "failed"
            row["error"] = str(exc)[:200]
        rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{ok}/{len(rows)} subjects processed successfully")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()