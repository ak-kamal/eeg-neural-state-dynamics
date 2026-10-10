"""ICLabel category distribution across a subject subset.

Produces the table cited in the README's "Key Design Decisions" section,
plus a full ICA-topography figure per subject (excluded components shown in
red). This is the diagnostic that motivated the notch-before-ICLabel choice
and the full-rank ICA choice: it lets us compare the resulting distribution
against published benchmarks (Pion-Tonachini et al. 2019, EEGLAB).

Caching: ICA fits and ICLabel outputs are cached under
data/derivatives/<subject>/cache/ using the same filenames as
notebooks/02_preprocessing_qc.ipynb. If the notebook or a previous run of
this script has already processed a subject, this script reuses those
artifacts. If a pipeline parameter has changed, delete the cache directory
first to force a refit.

Outputs:
    results/metrics/subset_qc_iclabel.json      per-subject category counts
    results/figures/<sub>_ica_topographies_*.png  per-subject topographies

Usage:
    uv run python scripts/subset_qc_summary.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive; no figure windows pop up
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mne

from eeg_state import config
from eeg_state import preprocessing as pp
from eeg_state.io import load_raw

SUBJECTS = ["sub-04", "sub-05", "sub-10"]
OUT_JSON = config.RESULTS / "metrics" / "subset_qc_iclabel.json"
OUT_FIG_DIR = config.RESULTS / "figures"


def _cache_dir(sub: str) -> Path:
    d = config.DATA_DERIV / sub / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_ica(sub: str, ica_copy: mne.io.BaseRaw) -> mne.preprocessing.ICA:
    """Load cached ICA for this subject, or fit and cache it."""
    cache = _cache_dir(sub) / f"{sub}_ica.fif"
    if cache.exists():
        print(f"  loading cached ICA from {cache}")
        return mne.preprocessing.read_ica(cache, verbose="ERROR")
    print("  fitting ICA (first run for this subject)")
    ica = pp.fit_ica(ica_copy)
    ica.save(cache, overwrite=True, verbose="ERROR")
    return ica


def get_iclabel(
    sub: str,
    ica: mne.preprocessing.ICA,
    ica_copy: mne.io.BaseRaw,
) -> tuple[list[str], list[int]]:
    """Load cached ICLabel output, or run and cache it."""
    cache = _cache_dir(sub) / f"{sub}_iclabel.json"
    if cache.exists():
        print(f"  loading cached ICLabel from {cache}")
        with open(cache) as f:
            data = json.load(f)
        return data["labels"], data["exclude"]
    print("  running ICLabel")
    labels_dict, exclude = pp.classify_ica_components(ica, ica_copy)
    with open(cache, "w") as f:
        json.dump(
            {
                "labels": labels_dict["labels"],
                "probas": labels_dict["y_pred_proba"].tolist(),
                "exclude": exclude,
            },
            f,
        )
    return labels_dict["labels"], exclude


def save_topographies(sub: str, ica: mne.preprocessing.ICA) -> None:
    """Save full ICA topographies, with excluded components reddened.

    Requires ica.exclude to be set before calling so that plot_components
    marks the excluded components in red.
    """
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = ica.plot_components(
        picks=list(range(ica.n_components_)),
        show=False,
        title=(
            f"{sub} — ICA topographies "
            f"(red = excluded, {len(ica.exclude)}/{ica.n_components_})"
        ),
    )
    fig_list = figs if isinstance(figs, list) else [figs]
    for n, fig in enumerate(fig_list):
        out = OUT_FIG_DIR / f"{sub}_ica_topographies_{n}.png"
        fig.set_size_inches(14, 10)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out}")


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    for sub in SUBJECTS:
        print(f"=== {sub} ===")
        raw = pp.highpass(load_raw(sub, preload=True))
        raw = pp.set_average_reference(raw)
        ica_copy = pp.prepare_ica_copy(raw, bad_channels=[])

        ica = get_ica(sub, ica_copy)
        labels, exclude = get_iclabel(sub, ica, ica_copy)
        ica.exclude = list(exclude)

        counts = Counter(labels)
        total = len(labels)
        results[sub] = {
            "n_components": total,
            "counts": dict(counts),
            "percentages": {k: 100.0 * v / total for k, v in counts.items()},
            "n_excluded": len(exclude),
        }

        for cat, n in counts.most_common():
            print(f"  {cat:15s}: {n:3d}  ({100 * n / total:.1f}%)")
        print(f"  excluded: {len(exclude)}")

        save_topographies(sub, ica)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()