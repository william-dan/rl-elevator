"""eventplot_log.py
A small CLI & importable helper to compare elevator‑event logs.

Usage
-----
$ python eventplot_log.py run1.txt run2.txt ...

Each *column* (x‑axis) represents one log file.  All events inside the file are
stacked in that column and colour‑coded by event type so you can scan across
runs and spot divergences at a glance.
The y‑axis is the absolute timestamp in seconds (earliest at the *top*).

Fixes in this version
---------------------
* **Parser** – now recognises that the timestamp and the `event:` keyword live
  on *different lines*.  It remembers the most recent `t=…` line and then pairs
  the next `event:` line with that time.
* **Orientation** – switched `eventplot` to **vertical** so time correctly maps
  to the y‑axis.
* Robust legend & colour mapping (extend the `EVENT_COLORS` dict as needed).
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Edit / extend this mapping if your logs contain additional event keywords.
# ---------------------------------------------------------------------------
EVENT_COLORS = {
    "CLOSE": "red",
    "OPEN": "green",
    "SPAWN": "orange",
}

# Pre‑compiled regexes are faster when parsing many lines.
_TIME_RE = re.compile(r"t=([0-9]+\.?[0-9]*)s")
_EVENT_RE = re.compile(r"event:\s*([A-Z]+)")

# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------

def parse_log(path: os.PathLike | str) -> List[Tuple[float, str]]:
    """Return a list of *(time_in_seconds, EVENT_NAME)* tuples for **path**.

    The log format looks like::
        t=3.4s | waiting=6
        event: SPAWN
        ...
        ----------------------------------------
    so the timestamp and event keyword appear on *different* lines.  We track
    the most recent timestamp and pair it with the next event line.
    """
    events: List[Tuple[float, str]] = []
    current_time: float | None = None

    with Path(path).expanduser().open(encoding="utf-8") as fh:
        for line in fh:
            mt = _TIME_RE.search(line)
            if mt:
                current_time = float(mt.group(1))
                continue  # look at next line for the event

            me = _EVENT_RE.search(line)
            
            if me and current_time is not None:
                # if me.group(1) == "SPAWN":
                events.append((current_time, me.group(1)))
                # we *do not* reset current_time – a block can have multiple
                # event lines (open/close pairs, etc.) with the same timestamp.

    # Sort just in case the file isn’t strictly ordered
    events.sort(key=lambda pair: pair[0])
    # print(f"events: {events}")
    return events

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_eventflows(log_paths: List[str]) -> None:
    """Create the comparison plot for *log_paths* and show it."""
    if not log_paths:
        raise ValueError("At least one log file must be provided.")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Iterate over the given files, creating one column (x‑offset) per file.
    for file_index, log_path in enumerate(log_paths):
        parsed = parse_log(log_path)

        if not parsed:
            print(f"[WARN] No events parsed from {log_path} – check the format?")
            continue

        # Bucket timestamps per event so we can colour‑code them.
        by_event: DefaultDict[str, List[float]] = defaultdict(list)
        for ts, ev in parsed:
            by_event[ev].append(ts)

        # Draw one *eventplot* per event‑type so we can assign different colours
        # – but all share the same *x* offset (== file_index), thereby merging
        # them into one logical column for the file.
        for ev, times in by_event.items():
            ax.eventplot(
                times,
                lineoffsets=file_index,
                linelengths=0.8,
                colors=EVENT_COLORS.get(ev, "black"),
                linewidths=2,
                orientation="vertical",  # puts time on y‑axis
            )

    # ──────────────────────────  Axes & Cosmetics  ──────────────────────────
    # Invert y so earlier timestamps appear on top.
    ax.invert_yaxis()

    ax.set_xlabel("Log File")
    ax.set_ylabel("Time (s)")

    # Label the x positions with the *basename* of each log.
    ax.set_xticks(range(len(log_paths)))
    ax.set_xticklabels([Path(p).name for p in log_paths], rotation=25, ha="right")

    ax.set_title("Elevator Event Timeline Comparison")

    # Build custom legend using colour mapping.
    legend_handles = [
        mlines.Line2D([], [], color=c, marker="|", markersize=15, linewidth=2, label=ev)
        for ev, c in EVENT_COLORS.items()
    ]
    ax.legend(handles=legend_handles, title="Event Type")

    # Light horizontal grid to follow the timeline easier.
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    plt.savefig("bad_3_1.png", dpi=300)
    plt.show()

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise event timelines from one or more elevator logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("logs", nargs="+", help="Path(s) to .txt log files")
    args = parser.parse_args()

    plot_eventflows(args.logs)


if __name__ == "__main__":  # pragma: no cover
    main()
