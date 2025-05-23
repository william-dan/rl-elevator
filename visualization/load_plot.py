#!/usr/bin/env python3
"""eventplot_open.py
====================
Visualise **only OPEN events** across any number of elevator logs and shade each
tick by the *load* inside the car (darker ⇒ fuller).

Usage
-----
$ python eventplot_open.py run_A.txt run_B.txt run_C.txt

* One **column** per file (x‑axis).  
* **Y‑axis = absolute time** (earliest at the top).  
* A **Greens** colour‑scale encodes passenger load and a colour‑bar shows the
  mapping.

The script works both as a CLI and as a library:

```python
from eventplot_open import build_plot
build_plot(["run_A.txt", "run_B.txt"])
```

Internals
~~~~~~~~~
* Robust two‑stage parser:
  1. Capture the latest `t=..s` timestamp.  
  2. When it encounters `event: OPEN`, it searches the following block for
     `load=` – that value is paired with the remembered timestamp.
* Global min/max load is used for normalisation so colours are comparable
  across files.
* Uses the modern `matplotlib.colormaps.get_cmap` API to avoid the deprecation
  warning.

Feel free to tweak the colormap (just change the name at the top), line
thickness (`linewidths=2`), or tick length (`linelengths=0.8`).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ── Regex helpers ────────────────────────────────────────────────────────────
TIME_RE = re.compile(r"^t=([0-9]+(?:\.[0-9]+)?)s")
EVENT_OPEN_RE = re.compile(r"^event:\s+OPEN")
LOAD_RE = re.compile(r"load=(\d+)")
SEPARATOR_RE = re.compile(r"^-{5,}")  # line of dashes

# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_open_events(path: Path) -> List[Tuple[float, int]]:
    """Return list of (timestamp, load) tuples for every OPEN event."""
    events: List[Tuple[float, int]] = []
    current_time: float | None = None

    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # 1) Timestamp line — remember it
        if (m_time := TIME_RE.match(line)):
            current_time = float(m_time.group(1))
            i += 1
            continue

        # 2) OPEN event — look ahead for load in the same block
        if EVENT_OPEN_RE.match(line):
            load_val: int | None = None
            j = i + 1
            while j < n and not SEPARATOR_RE.match(lines[j]):
                if (m_load := LOAD_RE.search(lines[j])):
                    load_val = int(m_load.group(1))
                    break
                j += 1

            if load_val is not None and current_time is not None:
                events.append((current_time, load_val))
            # jump to end of this block (either separator or next line)
            i = j
        i += 1
    return events

# ── Plotting ─────────────────────────────────────────────────────────────────

def build_plot(log_paths: List[Path]):
    """Create the eventplot figure from the provided log files."""
    datasets = [parse_open_events(p) for p in log_paths]

    # Flatten loads to compute global colour scale
    all_loads = [load for data in datasets for (_, load) in data]
    if not all_loads:
        raise SystemExit("No OPEN events found in the supplied logs.")

    vmin, vmax = min(all_loads), max(all_loads)
    if vmin == vmax:  # avoid zero‑range normalisation
        vmax += 1

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap("Greens")  # modern, no deprecation warning

    # ── Figure boilerplate ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(4, len(log_paths) * 3), 6))
    ax.set_title("OPEN Events – Load Intensity")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(range(len(log_paths)))
    ax.set_xticklabels([p.name for p in log_paths], rotation=45, ha="right")
    ax.invert_yaxis()  # earliest at the top
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    # ── Draw events ───────────────────────────────────────────────────────
    for col_idx, (path, events) in enumerate(zip(log_paths, datasets)):
        x_off = col_idx
        for t, load in events:
            c = cmap(norm(load))
            # Draw each tick individually so we can colour‑code per event
            ax.eventplot(
                [t],
                lineoffsets=x_off,
                linelengths=0.8,
                colors=[c],
                orientation="vertical",
                linewidths=2,
            )

    # ── Shared colour‑bar ────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # only needed for older mpl
    fig.colorbar(sm, ax=ax, label="Load (passengers)")

    fig.tight_layout()
    plt.savefig("load_6_1.png", dpi=300)
    plt.show()

# ── CLI ─────────────────────────────────────────────────────────────────────

def _main(argv: List[str] | None = None):  # noqa: D401
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot OPEN event timelines coloured by car load.",
    )
    parser.add_argument(
        "logs",
        metavar="LOG",
        type=Path,
        nargs="+",
        help="Path(s) to elevator log files",
    )
    args = parser.parse_args(argv)

    build_plot(args.logs)


if __name__ == "__main__":  # pragma: no cover
    _main(sys.argv[1:])
