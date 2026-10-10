"""Plot ICLabel-classified channel-noise components for selected subjects.

Complements the QC notebook by showing only the components whose
classification would indicate a bad channel. Used to test whether the
channel-noise labels are focal (single-electrode topography, consistent
with a bad channel) or distributed (consistent with misclassification).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_state import preprocessing as pp
from eeg_state.io import load_raw

SUBJECTS = ["sub-05", "sub-10"]


for sub in SUBJECTS:
    print(f"=== {sub} ===")
    raw = pp.highpass(load_raw(sub, preload=True))
    raw = pp.set_average_reference(raw)
    ica_copy = pp.prepare_ica_copy(raw, bad_channels=[])
    ica = pp.fit_ica(ica_copy)
    labels, _ = pp.classify_ica_components(ica, ica_copy)

    cats = labels["labels"]
    probas = labels["y_pred_proba"]

    noise_idx = [i for i, c in enumerate(cats) if c == "channel noise"]
    print(f"channel-noise components: {noise_idx}")
    print(f"their probabilities: {[round(float(probas[i]), 3) for i in noise_idx]}")

    if not noise_idx:
        print("  (none)")
        continue

    figs = ica.plot_components(
        picks=noise_idx,
        show=False,
        title=f"{sub} — channel-noise components",
    )
    if isinstance(figs, list):
        for f in figs:
            f.set_size_inches(10, 6)
            plt.show()
    else:
        figs.set_size_inches(10, 6)
        plt.show()

    # Also print the mean correlation of each component's strongest electrode
    # with its neighbours — a focal component should show one electrode with
    # disproportionately large weight.
    for i in noise_idx:
        topo = ica.get_components()[:, i]
        top_chan_idx = int(np.argmax(np.abs(topo)))
        top_chan = ica.ch_names[top_chan_idx]
        top_weight = float(topo[top_chan_idx])
        print(f"  ICA{i:03d}: strongest electrode = {top_chan} "
              f"(weight {top_weight:+.3f})")