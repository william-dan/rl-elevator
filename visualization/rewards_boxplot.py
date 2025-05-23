import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""Rewards Boxplot Utility

This script scans a folder for CSV files, looks for a column named
``reward`` in each, and visualises all the reward distributions together
in a **notched** box-and-whisker plot.  The plot is laid out with a clean
grid, mean diamonds, thicker median lines, larger figure size & higher
DPI so it displays crisply in notebooks and slides.

Usage (terminal or notebook)
---------------------------

.. code:: bash

   python rewards_boxplot.py                  # auto-scan /mnt/data
   python rewards_boxplot.py data/run_*.csv   # pass explicit files

You can also import :func:`plot_reward_boxplot` in your own notebooks.
"""


def load_reward_series(paths):
    """Return two lists: (data_series, labels) for a list of CSV paths."""
    series_list, labels = [], []
    for fp in paths:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print(f"⚠️  Could not read {fp}: {exc}")
            continue

        if 'Reward' not in df.columns:
            print(f"⚠️  Skipping {fp.name}: no 'reward' column.")
            continue

        series_list.append(df['Reward'].dropna())
        labels.append(fp.stem)
    return series_list, labels


def plot_reward_boxplot(data_series, labels, title="Reward distribution"):
    """Draw a notched boxplot with subtle style tweaks (no explicit colours)."""
    if not data_series:
        raise ValueError("No reward data provided.")

    plt.figure(figsize=(10, 6), dpi=120)
    boxprops   = dict(linestyle='-', linewidth=1.2)
    whiskerprops = dict(linestyle='--', linewidth=1.2)
    capprops   = dict(linewidth=1.2)
    medianprops = dict(linestyle='-', linewidth=2)
    meanprops  = dict(marker='D', markersize=6)  # ♦ mean diamond

    plt.boxplot(
        data_series,
        labels=labels,
        showmeans=True,
        notch=True,
        patch_artist=True,  # Respect default colour cycle
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        meanprops=meanprops,
    )

    ax = plt.gca()
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.tick_params(axis='x', rotation=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("rewards_boxplot_3_1.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    import sys

    # 1️⃣  Gather file paths (CLI args or wildcard scan) ----------------------
    if len(sys.argv) > 1:
        csv_files = [Path(p) for p in sys.argv[1:]]
    else:
        csv_files = list(Path('./Solver/').glob('*.csv'))

    # 2️⃣  Load data -----------------------------------------------------------
    series, labels = load_reward_series(csv_files)
    if not series:
        raise SystemExit("No valid 'reward' series found in the supplied CSV files.")

    # 3️⃣  Plot ---------------------------------------------------------------
    plot_reward_boxplot(series, labels)
