"""
day4_headway.py — DERIVE the real dispatching headway for TTC Route 29.

WHY THIS SCRIPT EXISTS (GAP 1)
------------------------------
`setup/ttc_route_29_data/dataloader.py` has a property `dispatching_headway`
that RETURNS A HARDCODED (300, 60) seconds — a guess ("5 min headway, 60 s
std").  The real headway measured from terminal departures is roughly TWICE
that: NORTH overall mean is ~536 s (median ~499 s, std ~295 s).  The 300 s
placeholder therefore makes buses dispatch about twice as often as reality,
which distorts the whole simulation (loads, bunching, holding pressure).

WHAT "DISPATCHING HEADWAY" MEANS HERE
-------------------------------------
The headway is the time gap between consecutive buses LEAVING the origin
terminal.  In the APC table the origin terminal is the row with StopSeq == 1
(NORTH terminal_start_id 11991 "Strachan Ave at Fleet St").  So:

    1. keep the direction's StopSeq == 1 rows (one per dispatched trip),
    2. parse StopDepartureTime into a real timestamp,
    3. within each SERVICE DATE, sort by time and take consecutive gaps,
    4. throw away absurd gaps (< 30 s duplicates, > 3600 s = a service break),
    5. summarise overall and by time-of-day PeriodID.

HOW TO RUN
----------
    cd /home/jiahao/Documents/busoperation
    python student_project/scripts/day4_headway.py

WHAT "DONE" LOOKS LIKE
----------------------
The printed NORTH overall mean is ~536 s (median ~499 s), and the by-period
means are roughly AM peak ~556, Midday ~500, PM peak ~548, Late evening ~689.
Outputs land in student_project/outputs/ (histogram, by-period bar chart, a
CSV and a JSON per direction).

This file is imported by build_env_data.py, which calls `derive_headway`.
`derive_headway(direction)` has NO side effects other than reading data.

See also:
    docs/03_data_to_env_mapping.md   (GAP 1, and GAP 5: lambda depends on this)
    scripts/day5_demand_od.py        (uses this headway to scale demand)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (path juggling must happen first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # headless: never needs a display
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# The value we are replacing — kept here so main() can print old-vs-new.
PLACEHOLDER_HEADWAY = (300.0, 60.0)  # (mean_sec, std_sec) from dataloader.py

# A consecutive-departure gap is only a real headway if it falls in this window.
# Below 30 s is almost always a duplicated/near-simultaneous scan; above 3600 s
# is a garage pull-out or an overnight gap, not a within-service headway.
MIN_HEADWAY_SEC = 30.0
MAX_HEADWAY_SEC = 3600.0

# A nice reading order for the time-of-day periods (only those present are used).
PERIOD_ORDER = [
    "Early morning", "Morning", "AM peak", "Midday",
    "Afternoon", "PM peak", "Early evening", "Late evening",
]


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def _terminal_departure_gaps(direction: str) -> pd.DataFrame:
    """Consecutive terminal-departure gaps for one direction.

    Returns a tidy DataFrame with one row per observed headway:
        headway_sec, PeriodID, DAYTYPE, date

    Each gap is labelled with the PeriodID / DAYTYPE of the *later* departure
    (the bus that just left), so a headway "belongs" to the moment it occurred.

    Reading only — safe to import and call.
    """
    apc = C.load_apc_clean()
    d = C.route_dir(apc, direction)

    # STUDENT TODO: Route 29 mixes branches (DLWI, 29Dcon, DLPRcon).  They all
    # dispatch from the same origin terminal, so we treat every StopSeq == 1
    # departure as one event regardless of branch (this is what a rider at the
    # terminal actually experiences).  If you later want branch-specific
    # headways, add a `.groupby("Branch")` here.
    d = d[d["StopSeq"] == 1].copy()

    d["dep"] = pd.to_datetime(d["StopDepartureTime"], errors="coerce")
    d = d.dropna(subset=["dep"])

    # Service DATE, so an overnight wrap never looks like a 6-hour headway.
    d["date"] = d["dep"].dt.date
    d = d.sort_values(["date", "dep"])

    # Gap to the previous departure on the same date, in seconds.
    d["headway_sec"] = d.groupby("date")["dep"].diff().dt.total_seconds()

    d = d[
        (d["headway_sec"] > MIN_HEADWAY_SEC)
        & (d["headway_sec"] < MAX_HEADWAY_SEC)
    ]
    return d[["headway_sec", "PeriodID", "DAYTYPE", "date"]].reset_index(drop=True)


def _summarize(values: np.ndarray) -> tuple[float, float, int]:
    """(mean, sample-std, n) for a 1-D array; std is 0 when n < 2."""
    n = int(values.size)
    if n == 0:
        return 0.0, 0.0, 0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n >= 2 else 0.0
    return mean, std, n


def derive_headway(direction: str) -> dict:
    """Real dispatching headway for one direction ('NORTH' or 'SOUTH').

    Returns
    -------
    dict with exactly:
        "overall"   : (mean_sec, std_sec)
        "by_period" : {PeriodID: (mean_sec, std_sec, n)}

    This is the function build_env_data.derive_all() consumes to replace the
    hardcoded (300, 60) placeholder (GAP 1).  No side effects but reading data.
    """
    gaps = _terminal_departure_gaps(direction)

    overall_mean, overall_std, _ = _summarize(gaps["headway_sec"].to_numpy())

    by_period: dict[str, tuple[float, float, int]] = {}
    for period, grp in gaps.groupby("PeriodID"):
        by_period[str(period)] = _summarize(grp["headway_sec"].to_numpy())

    return {"overall": (overall_mean, overall_std), "by_period": by_period}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_histogram(gaps: pd.DataFrame, direction: str, real_mean: float) -> None:
    """Headway distribution with the 300 s placeholder and real mean marked."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(gaps["headway_sec"], bins=40, color="#4c78a8",
            edgecolor="white", alpha=0.85)
    ax.axvline(PLACEHOLDER_HEADWAY[0], color="#e45756", linestyle="--",
               linewidth=2, label=f"placeholder {PLACEHOLDER_HEADWAY[0]:.0f}s")
    ax.axvline(real_mean, color="#54a24b", linestyle="-", linewidth=2,
               label=f"real mean {real_mean:.0f}s")
    ax.set_xlabel("Headway between terminal departures (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"TTC Route 29 {direction} — dispatching headway distribution")
    ax.legend()
    C.savefig(fig, f"day4_headway_hist_{direction.lower()}.png")
    plt.close(fig)


def _plot_by_period(by_period: dict, direction: str) -> None:
    """Bar chart of mean headway per time-of-day period (± std)."""
    periods = [p for p in PERIOD_ORDER if p in by_period]
    periods += [p for p in by_period if p not in periods]  # any unexpected ones
    means = [by_period[p][0] for p in periods]
    stds = [by_period[p][1] for p in periods]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(periods)), means, yerr=stds, capsize=4,
           color="#4c78a8", edgecolor="white")
    ax.axhline(PLACEHOLDER_HEADWAY[0], color="#e45756", linestyle="--",
               linewidth=2, label=f"placeholder {PLACEHOLDER_HEADWAY[0]:.0f}s")
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=30, ha="right")
    ax.set_ylabel("Mean headway (s)")
    ax.set_title(f"TTC Route 29 {direction} — headway by time of day")
    ax.legend()
    C.savefig(fig, f"day4_headway_by_period_{direction.lower()}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Reporting for one direction
# --------------------------------------------------------------------------- #
def _report_direction(direction: str) -> None:
    direction = direction.upper()
    C.banner(f"DAY 4 — DISPATCHING HEADWAY  ({direction})")

    gaps = _terminal_departure_gaps(direction)
    result = derive_headway(direction)
    mean_sec, std_sec = result["overall"]
    vals = gaps["headway_sec"].to_numpy()
    median_sec = float(np.median(vals)) if vals.size else 0.0

    # ---- old vs new -------------------------------------------------------- #
    print(f"\nDeparture events analysed : {len(gaps):,} headways "
          f"(from StopSeq == 1 rows across {gaps['date'].nunique()} service days)")
    print("\nPLACEHOLDER (dataloader.py) vs REAL (measured):")
    print(f"  mean : {PLACEHOLDER_HEADWAY[0]:7.0f} s   ->   {mean_sec:7.1f} s"
          f"   ({mean_sec / PLACEHOLDER_HEADWAY[0]:.2f}x the placeholder)")
    print(f"  std  : {PLACEHOLDER_HEADWAY[1]:7.0f} s   ->   {std_sec:7.1f} s")
    print(f"  median (real only)         :   {median_sec:7.1f} s")
    print("  -> the 300 s placeholder dispatches buses about twice too often.")

    # ---- by time-of-day period -------------------------------------------- #
    print("\nBy time-of-day period (PeriodID):")
    print(f"  {'period':<15}{'mean_s':>9}{'std_s':>9}{'n':>8}")
    ordered = [p for p in PERIOD_ORDER if p in result["by_period"]]
    ordered += [p for p in result["by_period"] if p not in ordered]
    for p in ordered:
        m, s, n = result["by_period"][p]
        print(f"  {p:<15}{m:>9.1f}{s:>9.1f}{n:>8,}")

    # ---- by day type (weekday / Sat / Sun) -------------------------------- #
    print("\nBy day type (DAYTYPE):")
    print(f"  {'daytype':<15}{'mean_s':>9}{'std_s':>9}{'n':>8}")
    by_daytype: dict[str, tuple[float, float, int]] = {}
    for dt, grp in gaps.groupby("DAYTYPE"):
        by_daytype[str(dt)] = _summarize(grp["headway_sec"].to_numpy())
        m, s, n = by_daytype[str(dt)]
        print(f"  {str(dt):<15}{m:>9.1f}{s:>9.1f}{n:>8,}")

    # STUDENT TODO — MODELLING DECISION:
    #   Should the RL environment run with ONE headway (this overall mean/std),
    #   or with TIME-OF-DAY-SPECIFIC scenarios (a different headway per period,
    #   e.g. AM peak vs Late evening)?  A single headway is simplest and matches
    #   the current DataLoader interface (one (mean, std) tuple).  Period-specific
    #   scenarios are more realistic — bunching pressure differs a lot between a
    #   500 s midday headway and a 689 s late-evening headway — but require the
    #   simulator to switch headway by simulated time.  Start with the overall
    #   value; graduate to per-period scenarios once the baseline works.

    # ---- plots ------------------------------------------------------------- #
    print()
    _plot_histogram(gaps, direction, mean_sec)
    _plot_by_period(result["by_period"], direction)

    # ---- tabular outputs (CSV + JSON) ------------------------------------- #
    rows = [{"group_kind": "overall", "group": "ALL",
             "mean_sec": mean_sec, "std_sec": std_sec, "n": int(vals.size)}]
    for p in ordered:
        m, s, n = result["by_period"][p]
        rows.append({"group_kind": "period", "group": p,
                     "mean_sec": m, "std_sec": s, "n": n})
    for dt, (m, s, n) in by_daytype.items():
        rows.append({"group_kind": "daytype", "group": dt,
                     "mean_sec": m, "std_sec": s, "n": n})
    C.save_csv(pd.DataFrame(rows), f"day4_headway_{direction.lower()}.csv")

    C.save_json(
        {
            "direction": direction,
            "placeholder": {"mean_sec": PLACEHOLDER_HEADWAY[0],
                            "std_sec": PLACEHOLDER_HEADWAY[1]},
            "overall": {"mean_sec": mean_sec, "median_sec": median_sec,
                        "std_sec": std_sec, "n": int(vals.size)},
            "by_period": {p: {"mean_sec": v[0], "std_sec": v[1], "n": v[2]}
                          for p, v in result["by_period"].items()},
            "by_daytype": {dt: {"mean_sec": v[0], "std_sec": v[1], "n": v[2]}
                           for dt, v in by_daytype.items()},
        },
        f"day4_headway_{direction.lower()}.json",
    )


def main() -> None:
    for direction in C.DIRECTIONS:
        _report_direction(direction)
    C.banner("DAY 4 DONE")
    print("Next: feed derive_headway() into scripts/day5_demand_od.py (lambda "
          "= boardings/trip / headway, GAP 5) and into build_env_data.py.")


if __name__ == "__main__":
    main()
