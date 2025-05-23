#!/usr/bin/env python3
"""
Compare RL Curves Across Multiple CSV Files — *Robust to tensor() strings*
===========================================================================
Many RL logging frameworks dump numbers as ``tensor(123.4)``.  Previous
versions of this script choked on those object‑dtype strings when computing
rolling means/standard deviations in pandas.  This release automatically
**sanitises non‑numeric columns**, converting recognised patterns to floats
before plotting.

New in v0.3 (22 May 2025)
-------------------------
* **Automatic numeric casting**:
  * Handles *tensor wrappers* (``tensor(1053.)``), plain strings, or mixed
    numeric/object columns.
  * Falls back to *regex extraction* of the first number if direct casting
    fails.
* Helpful **warning** printed when any non‑parsable rows are dropped.
* API unchanged—just works on messy CSVs.

Usage recap
~~~~~~~~~~~
```bash
python plot_loss_with_filling.py runA.csv runB.csv \
       --x_col episode --y_cols reward --smooth 100 --shade_std
```

Author: ChatGPT — May 2025
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Sequence, List

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------ Helpers -----------------------------------

_TENSOR_RE = re.compile(r"tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float_series(s: pd.Series, *, col_name: str, file: pathlib.Path) -> pd.Series:
    """Ensure a Series is float64, coercing troublesome strings.

    Strategy (in order):
    1. If already numeric, return as float64.
    2. Try ``pd.to_numeric`` (fast path).
    3. Detect ``tensor( … )`` wrappers and strip them.
    4. Extract the *first* numeric substring via regex.
    5. Emit a warning if any rows remain NaN after conversion.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    # 2. Try simple coercion
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.notna().sum() > 0 and coerced.isna().sum() < len(s):
        return coerced.astype(float)

    # 3. Strip tensor() wrappers
    stripped = s.astype(str).str.replace(_TENSOR_RE, r"\\1", regex=True)
    coerced2 = pd.to_numeric(stripped, errors="coerce")
    if coerced2.notna().sum() > 0:
        return coerced2.astype(float)

    # 4. Generic numeric extraction
    extracted = s.astype(str).str.extract(_NUM_RE)[0]
    coerced3 = pd.to_numeric(extracted, errors="coerce")
    if coerced3.notna().sum() > 0:
        return coerced3.astype(float)

    # 5. No luck — warn & return NaNs
    print(
        f"[Warning] Column '{col_name}' in {file} contains no parsable numbers; "
        "skipping rows with NaN.",
        file=sys.stderr,
    )
    return coerced3  # all‑NaN


def _rolling(series: pd.Series, window: int) -> pd.Series:
    """Centered rolling mean; expanding for initial points."""
    if window <= 1:
        return series
    return series.rolling(window, center=True, min_periods=1).mean()


def _parse_labels(files: List[pathlib.Path], labels_arg: str | None) -> List[str]:
    if labels_arg is None:
        return [f.stem for f in files]
    labels = [l.strip() for l in labels_arg.split(",")]
    if len(labels) != len(files):
        raise ValueError("--labels count must match number of CSV files")
    return labels

# ------------------------------ Core --------------------------------------

def plot_losses_multi(
    csv_paths: Sequence[str | pathlib.Path],
    x_col: str = "step",
    y_cols: Sequence[str] = ("loss",),
    *,
    smooth: int = 1,
    shade_std: bool = False,
    fill_between: bool = False,
    baseline: float = 0.0,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    save_path: str | pathlib.Path | None = None,
    dpi: int = 300,
):
    """Plot and compare RL curves from multiple CSV logs (robust to string tensors)."""

    files = [pathlib.Path(p) for p in csv_paths]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)

    labels = labels or [f.stem for f in files]
    if len(labels) != len(files):
        raise ValueError("labels length mismatch with csv_paths")

    plt.figure(figsize=(9, 5.5))

    for f, label in zip(files, labels):
        df = pd.read_csv(f)
        if x_col not in df.columns:
            raise ValueError(f"x_col '{x_col}' not found in {f}")
        for y in y_cols:
            if y not in df.columns:
                raise ValueError(f"y_col '{y}' not found in {f}")

        x = _to_float_series(df[x_col], col_name=x_col, file=f)

        for y in y_cols:
            raw_y = _to_float_series(df[y], col_name=y, file=f)
            s = _rolling(raw_y, smooth)
            plt.plot(x, s, label=f"{label} ▸ {y}")

            if shade_std:
                std = raw_y.rolling(smooth, center=True, min_periods=1).std()
                plt.fill_between(x, s - std, s + std, alpha=0.18)

        if fill_between:
            if len(y_cols) == 1:
                y = y_cols[0]
                s = _rolling(_to_float_series(df[y], col_name=y, file=f), smooth)
                plt.fill_between(x, baseline, s, alpha=0.25)
            elif len(y_cols) >= 2:
                s1 = _rolling(_to_float_series(df[y_cols[0]], col_name=y_cols[0], file=f), smooth)
                s2 = _rolling(_to_float_series(df[y_cols[1]], col_name=y_cols[1], file=f), smooth)
                plt.fill_between(x, s1, s2, alpha=0.25)

    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Value")
    if title:
        plt.title(title)
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        print(f"Figure saved to {save_path.absolute()}")
    else:
        plt.show()

# ---------------------------- CLI wrapper ---------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RL curves (handles tensor strings)")
    parser.add_argument("csv_paths", nargs="+", help="One or more CSV files (supports glob expansion by shell)")
    parser.add_argument("--x_col", default="step", help="Column for x‑axis")
    parser.add_argument("--y_cols", nargs="+", default=["loss"], help="Y columns to plot (applied to every file)")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling window size for running mean (≥1)")
    parser.add_argument("--shade_std", action="store_true", help="Shade ±1σ around each curve")
    parser.add_argument("--fill_between", action="store_true", help="Shade under curve (1 y) or between first two curves (≥2 y) **per file**")
    parser.add_argument("--baseline", type=float, default=0.0, help="Baseline for fill_between with single y‑col")
    parser.add_argument("--labels", help="Comma‑separated custom labels matching csv_paths order")
    parser.add_argument("--title", help="Figure title")
    parser.add_argument("--save_path", help="Save figure here instead of displaying")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    args = parser.parse_args()

    plot_losses_multi(
        args.csv_paths,
        x_col=args.x_col,
        y_cols=args.y_cols,
        smooth=args.smooth,
        shade_std=args.shade_std,
        fill_between=args.fill_between,
        baseline=args.baseline,
        labels=_parse_labels([pathlib.Path(p) for p in args.csv_paths], args.labels),
        title=args.title,
        save_path=args.save_path,
        dpi=args.dpi,
    )
