"""
day1_explore.py — Day 1 orientation / exploratory data analysis (EDA).

GOAL FOR TODAY
--------------
Get your hands dirty with the three raw data tables you will spend the week
turning into a simulation environment. You do NOT derive anything final today —
you just *look*. By the end you should be able to answer:

    * How big are the tables, and what time span do they cover?
    * What routes / directions / branches / periods / day-types live inside?
    * What is the canonical NORTH stop sequence (terminal -> terminal)?
    * Roughly how many people board/alight, and how loaded are the buses?
    * What does the REAL dispatching headway look like at the terminal, and how
      badly does the current hard-coded placeholder (300 s) miss it? (This is
      GAP 1 — see docs/03_data_to_env_mapping.md.)
    * Is APC (2023) our geometry source of truth, while the GTFS feed is only a
      2020 cross-check? (GAP 4 — the provenance mismatch, proven with numbers.)

WHAT THIS SCRIPT PRODUCES
-------------------------
Printed report (read it top to bottom) plus, in student_project/outputs/:
    day1_boardings_per_stop_north.png     (a) boardings along the route
    day1_load_profile_north.png           (b) average arrival/departure load
    day1_headway_hist_north.png           (c) terminal headways, 300 s marked
    day1_avl_kph_hist_north.png           (d) instantaneous GPS speed (KPH)
    day1_summary.json                     the headline numbers

HOW TO RUN (from the repo root /home/jiahao/Documents/busoperation):
    python student_project/scripts/day1_explore.py

WHAT "DONE" LOOKS LIKE
----------------------
The script finishes in well under a minute, the four PNGs and day1_summary.json
appear in student_project/outputs/, and the printed headway is clearly ~500+ s,
NOT 300 s. Next: docs/04_week_plan.md (Day 2) and scripts/day2_geometry.py.

This file is Day-1 teaching code: read the comments, they explain *why*, not
just *what*. It intentionally does NOT expose derive_* functions — that starts
on Day 2.
"""

from __future__ import annotations

# --- Standard "every script starts like this" preamble ---------------------- #
import sys
from pathlib import Path

# Make `import common as C` work no matter what directory you launch from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (import after sys.path tweak — this is intentional)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Matplotlib must be told to use a non-interactive ("Agg") backend BEFORE
# pyplot is imported, otherwise it will try to open a window and crash on a
# headless machine / server.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# We explore NORTH today. SOUTH is structurally symmetric — the exact same code
# works by flipping this one constant (the loaders/helpers all take a direction
# argument). On a later day you will loop over C.DIRECTIONS in build_env_data.py.
#
# STUDENT TODO: once you are comfortable, turn DIRECTION into an argparse flag
# (e.g. `--direction SOUTH`) and re-run to eyeball the southbound numbers.
DIRECTION = "NORTH"

# The placeholder the current simulator ships with. Everything we measure below
# gets contrasted against these so you can *see* how wrong the placeholder is.
PLACEHOLDER_HEADWAY_SEC = 300.0   # GAP 1: hard-coded (mean=300, std=60) seconds
PLACEHOLDER_HEADWAY_STD = 60.0

# Sanity filters for terminal headways (matches what Day 4 will use so the
# preview number and the final number line up). A gap under 30 s is a duplicate
# ping; a gap over 3600 s is a service break / overnight gap, not a headway.
HEADWAY_MIN_SEC = 30.0
HEADWAY_MAX_SEC = 3600.0

# For the speed histogram we drop obviously-bogus GPS speeds. Real transit buses
# on an arterial top out well under this; anything above is a data glitch.
KPH_MAX_PLOT = 80.0


# --------------------------------------------------------------------------- #
# Small helpers (Day-1 only — nothing here is imported by other scripts)
# --------------------------------------------------------------------------- #
def _terminal_headways(apc_clean: pd.DataFrame, direction: str) -> np.ndarray:
    """Preview of the REAL dispatching headway (seconds).

    Method (kept deliberately simple — Day 4 formalises it in derive_headway):
      1. Keep only terminal *departures*: the rows with StopSeq == 1.
      2. Parse StopDepartureTime to a real timestamp.
      3. Within each service date, sort by time and take successive gaps.
      4. Drop non-physical gaps (< 30 s duplicates, > 3600 s service breaks).

    We group by calendar date so an overnight gap between the last bus of one
    day and the first bus of the next is never mistaken for a headway.
    """
    d = C.route_dir(apc_clean, direction)

    # Terminal departures only. StopSeq == 1 is the first stop of the trip.
    term = d[d["StopSeq"] == 1].copy()

    # Turn the string timestamps into real datetimes. Bad/blank values become
    # NaT and are dropped.
    term["dep"] = pd.to_datetime(term["StopDepartureTime"], errors="coerce")
    term = term.dropna(subset=["dep"])

    # Service date = the calendar date of the departure (good enough here; the
    # NORTH terminal buses depart mid-day/evening, not across midnight).
    term["service_date"] = term["dep"].dt.date

    gaps = []
    for _, day in term.groupby("service_date"):
        times = day["dep"].sort_values()
        # .diff() gives consecutive gaps as Timedelta; convert to seconds.
        diffs = times.diff().dropna().dt.total_seconds().to_numpy()
        gaps.append(diffs)

    if not gaps:
        return np.array([])

    all_gaps = np.concatenate(gaps)
    # Keep only physically-plausible headways.
    mask = (all_gaps >= HEADWAY_MIN_SEC) & (all_gaps <= HEADWAY_MAX_SEC)
    return all_gaps[mask]


def _per_stop_profile(apc_clean: pd.DataFrame, seq: pd.DataFrame,
                      direction: str) -> pd.DataFrame:
    """Per-stop boarding/alighting/load table, ordered along the route.

    Returns the canonical stop sequence (`seq`) with these columns bolted on:
        total_boardings, total_alightings, mean_arrload, mean_depload, n_trips
    Stops are in terminal->terminal order because `seq` already is.
    """
    d = C.route_dir(apc_clean, direction)
    d = d.copy()
    d["StopID"] = d["StopID"].astype(str)

    agg = (
        d.groupby("StopID")
        .agg(
            total_boardings=("Boarding", "sum"),
            total_alightings=("Alighting", "sum"),
            mean_arrload=("Arrload", "mean"),
            mean_depload=("Depload", "mean"),
        )
        .reset_index()
    )

    # Left-join onto the ordered sequence so plotting order == route order.
    prof = seq.merge(agg, on="StopID", how="left")
    return prof


def _short_stop_label(row: pd.Series) -> str:
    """A compact human label for a stop, e.g. 'Dufferin St @ Wilson Ave'."""
    on = str(row.get("ONSTREET", "") or "").strip()
    at = str(row.get("ATSTREET", "") or "").strip()
    if on and at:
        return f"{on} @ {at}"
    return on or at or str(row["StopID"])


def _provenance_check(apc_clean: pd.DataFrame, apc_raw: pd.DataFrame,
                      direction: str) -> dict:
    """GAP 4 evidence — is APC (2023) the source of truth, GTFS (2020) only a check?

    Computes three *reproducible* facts (so the student never has to take the
    72% figure on faith — this script regenerates it):
      * APC per-stop Lat/Lon coverage for this direction (expected ~100%): this
        is why Day 2 derives geometry from APC, not GTFS.
      * the fraction of 2023 APC Route-29 stop_ids that ALSO appear in the 2020
        GTFS stops.txt (expected ~72%): the provenance mismatch itself.
      * the GTFS calendar.txt service-date span (2020) — contrast it with the
        APC observation window (Nov 2023) printed by the caller.
    """
    # (1) APC Lat/Lon coverage — the geometry source used on Day 2.
    ar = C.route_dir(apc_raw, direction)
    if {"Lat", "Lon"}.issubset(ar.columns) and len(ar):
        latlon_cov = float(ar[["Lat", "Lon"]].notna().all(axis=1).mean())
    else:
        latlon_cov = float("nan")

    # (2) 2023 APC Route-29 stop set vs the 2020 GTFS stop set.
    apc_ids = set(apc_clean.loc[apc_clean["Route"] == C.ROUTE, "StopID"].astype(str))
    gtfs_stops = C.load_gtfs("stops")
    gtfs_ids = set(gtfs_stops["stop_id"].astype(str))
    overlap = len(apc_ids & gtfs_ids) / len(apc_ids) if apc_ids else float("nan")

    # (3) GTFS calendar service window (these dates are 2020, not 2023).
    cal = C.load_gtfs("calendar")
    cal_start = str(int(cal["start_date"].min()))
    cal_end = str(int(cal["end_date"].max()))

    return {
        "apc_latlon_coverage": latlon_cov,
        "gtfs_stop_overlap": overlap,
        "n_apc_route29_stops": len(apc_ids),
        "n_gtfs_stops": len(gtfs_ids),
        "gtfs_calendar_start": cal_start,
        "gtfs_calendar_end": cal_end,
    }


# --------------------------------------------------------------------------- #
# Plotting (each returns nothing; it saves via C.savefig)
# --------------------------------------------------------------------------- #
def plot_boardings_per_stop(prof: pd.DataFrame, direction: str) -> None:
    """(a) Total boardings at each stop, in route order."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(prof))
    ax.bar(x, prof["total_boardings"].fillna(0.0), color="#2a6f97")
    ax.set_xlabel(f"Stop index along route  (0 = terminal start, "
                  f"{len(prof) - 1} = terminal end)")
    ax.set_ylabel("Total boardings (whole month)")
    ax.set_title(f"Route {C.ROUTE} {direction}: boardings per stop")
    ax.margins(x=0.01)
    C.savefig(fig, f"day1_boardings_per_stop_{direction.lower()}.png")
    plt.close(fig)


def plot_load_profile(prof: pd.DataFrame, direction: str) -> None:
    """(b) Average on-board load (arrival & departure) along the route."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(prof))
    ax.plot(x, prof["mean_arrload"], marker="o", ms=3, lw=1.4,
            label="mean Arrload (load on arrival)", color="#e07a5f")
    ax.plot(x, prof["mean_depload"], marker="s", ms=3, lw=1.4,
            label="mean Depload (load on departure)", color="#3d5a80")
    ax.set_xlabel("Stop index along route")
    ax.set_ylabel("Mean passengers on board")
    ax.set_title(f"Route {C.ROUTE} {direction}: average load profile")
    ax.legend()
    ax.margins(x=0.01)
    C.savefig(fig, f"day1_load_profile_{direction.lower()}.png")
    plt.close(fig)


def plot_headway_hist(headways: np.ndarray, direction: str) -> None:
    """(c) Histogram of terminal headways, with the 300 s placeholder marked."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if headways.size:
        ax.hist(headways, bins=40, color="#81b29a", edgecolor="white")
        real_mean = float(np.mean(headways))
        real_median = float(np.median(headways))
        ax.axvline(real_mean, color="#3d405b", lw=2,
                   label=f"real mean = {real_mean:.0f} s")
        ax.axvline(real_median, color="#3d405b", lw=2, ls=":",
                   label=f"real median = {real_median:.0f} s")
    ax.axvline(PLACEHOLDER_HEADWAY_SEC, color="#e63946", lw=2, ls="--",
               label=f"placeholder = {PLACEHOLDER_HEADWAY_SEC:.0f} s (GAP 1)")
    ax.set_xlabel("Terminal-to-terminal headway (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(f"Route {C.ROUTE} {direction}: dispatching headway "
                 f"(real vs. placeholder)")
    ax.legend()
    C.savefig(fig, f"day1_headway_hist_{direction.lower()}.png")
    plt.close(fig)


def plot_kph_hist(kph: np.ndarray, direction: str) -> None:
    """(d) Histogram of instantaneous GPS speed from the AVL pings."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if kph.size:
        ax.hist(kph, bins=40, color="#9d8189", edgecolor="white")
        ax.axvline(float(np.mean(kph)), color="#3d405b", lw=2,
                   label=f"mean = {np.mean(kph):.1f} km/h")
    ax.set_xlabel(f"Instantaneous speed KPH (0..{KPH_MAX_PLOT:.0f}; "
                  f"the 0-spike = dwelling at stops)")
    ax.set_ylabel("Ping count")
    ax.set_title(f"Route {C.ROUTE} {direction}: AVL instantaneous speed")
    ax.legend()
    C.savefig(fig, f"day1_avl_kph_hist_{direction.lower()}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    direction = DIRECTION

    # ----------------------------------------------------------------------- #
    # 1) Load the three tables. These are a few hundred-thousand rows each —
    #    fine to read fully; the whole script still finishes in seconds.
    # ----------------------------------------------------------------------- #
    C.banner(f"DAY 1 EDA — Route {C.ROUTE} {direction} "
             f"(SOUTH is symmetric: flip DIRECTION at the top of this file)")
    print("Loading raw tables via common.py ...")
    apc_clean = C.load_apc_clean()   # boarding/alighting/load/dwell + times
    apc_raw = C.load_apc_raw()       # same idea + per-stop Lat/Lon (geometry)
    avl_seg = C.load_avl_seg()       # cleaned GPS pings with KPH

    # ----------------------------------------------------------------------- #
    # 2) Shapes + what categorical values live inside each table.
    # ----------------------------------------------------------------------- #
    C.banner("Dataset shapes")
    print(f"  APC clean : {apc_clean.shape[0]:>8,} rows x {apc_clean.shape[1]} cols")
    print(f"  APC raw   : {apc_raw.shape[0]:>8,} rows x {apc_raw.shape[1]} cols "
          f"(has Lat/Lon)")
    print(f"  AVL seg   : {avl_seg.shape[0]:>8,} rows x {avl_seg.shape[1]} cols")

    # Date range from the APC arrival timestamps.
    arr = pd.to_datetime(apc_clean["StopArrivalTime"], errors="coerce")
    date_min, date_max = arr.min(), arr.max()
    print(f"\n  Date range (APC StopArrivalTime): {date_min}  ..  {date_max}")

    C.banner("Categorical values (APC clean)")
    for col in ("Route", "RouteDirection", "Branch", "PeriodID", "DAYTYPE"):
        vals = sorted(map(str, apc_clean[col].dropna().unique()))
        print(f"  {col:<15}: {vals}")

    # ----------------------------------------------------------------------- #
    # 2b) DATA PROVENANCE (GAP 4): APC (2023) is the source of truth; the GTFS
    #     feed is a 2020 snapshot and only a cross-check. Prove it with numbers.
    # ----------------------------------------------------------------------- #
    prov = _provenance_check(apc_clean, apc_raw, direction)
    C.banner("Data provenance & source-of-truth (GAP 4)")
    print(f"  APC per-stop Lat/Lon coverage ({direction}) : "
          f"{prov['apc_latlon_coverage'] * 100:5.1f}%   "
          f"(=> APC gives real geometry; used on Day 2)")
    print(f"  APC Route-29 unique stop_ids                : "
          f"{prov['n_apc_route29_stops']}")
    print(f"  ... also present in GTFS stops.txt          : "
          f"{prov['gtfs_stop_overlap'] * 100:5.1f}%   "
          f"(partial match -> GTFS is 2020, our data is 2023)")
    print(f"  GTFS calendar span (service dates)          : "
          f"{prov['gtfs_calendar_start']} .. {prov['gtfs_calendar_end']}  (2020!)")
    print(f"  APC observation window                      : "
          f"{date_min}  ..  {date_max}  (Nov 2023)")
    print("  --> Prefer APC Lat/Lon for geometry; treat GTFS shapes as an")
    print("      optional cross-check only. This is GAP 4.")

    # ----------------------------------------------------------------------- #
    # 3) The canonical stop sequence (terminal -> terminal) for this direction.
    #    common.stop_sequence() does the median-StopSeq ordering for us.
    # ----------------------------------------------------------------------- #
    seq = C.stop_sequence(apc_clean, direction)
    n_stops = len(seq)
    term_start = seq.iloc[0]
    term_end = seq.iloc[-1]

    C.banner(f"Canonical stop sequence: {n_stops} stops "
             f"(terminal -> terminal)")
    print("  First 5 stops:")
    for _, r in seq.head(5).iterrows():
        print(f"    seq {int(r['stop_seq']):>3}  id={r['StopID']:>7}  "
              f"{_short_stop_label(r)}  (n_trips={int(r['n_trips'])})")
    print("  ...")
    print("  Last 5 stops:")
    for _, r in seq.tail(5).iterrows():
        print(f"    seq {int(r['stop_seq']):>3}  id={r['StopID']:>7}  "
              f"{_short_stop_label(r)}  (n_trips={int(r['n_trips'])})")
    print(f"\n  TERMINAL START : id={term_start['StopID']}  "
          f"{_short_stop_label(term_start)}")
    print(f"  TERMINAL END   : id={term_end['StopID']}  "
          f"{_short_stop_label(term_end)}")
    # STUDENT TODO: compare these two terminal IDs against summary_north.json in
    # setup/ttc_route_29_data/ (expected 11991 -> 2108 for NORTH). They should
    # match; if not, a branch is skewing the median ordering (see common.py).

    # ----------------------------------------------------------------------- #
    # 4) Demand / load headlines.
    # ----------------------------------------------------------------------- #
    d = C.route_dir(apc_clean, direction)
    total_boardings = int(d["Boarding"].sum())
    total_alightings = int(d["Alighting"].sum())
    mean_arrload = float(d["Arrload"].mean())
    mean_depload = float(d["Depload"].mean())
    mean_dwell = float(d["AvgDwell"].mean())
    n_trips = int(d["TripID"].nunique())

    C.banner("Demand & load headlines")
    print(f"  Unique trips (month)     : {n_trips:,}")
    print(f"  Total boardings          : {total_boardings:,}")
    print(f"  Total alightings         : {total_alightings:,}")
    print(f"  Mean load on arrival     : {mean_arrload:.2f} pax")
    print(f"  Mean load on departure   : {mean_depload:.2f} pax")
    print(f"  Mean dwell time          : {mean_dwell:.1f} s")
    # In a closed line, monthly boardings ~ monthly alightings (everyone who
    # gets on eventually gets off). A big mismatch hints at data issues.
    if total_alightings:
        print(f"  boardings / alightings   : "
              f"{total_boardings / total_alightings:.3f}  (expect ~1.0)")

    # Build the per-stop profile used by plots (a) and (b).
    prof = _per_stop_profile(apc_clean, seq, direction)

    # ----------------------------------------------------------------------- #
    # 5) THE HEADLINE: real terminal headway vs. the 300 s placeholder (GAP 1).
    # ----------------------------------------------------------------------- #
    headways = _terminal_headways(apc_clean, direction)
    C.banner("Dispatching headway preview — REAL vs. PLACEHOLDER (GAP 1)")
    if headways.size:
        h_mean = float(np.mean(headways))
        h_median = float(np.median(headways))
        h_std = float(np.std(headways))
        print(f"  measured from {headways.size:,} terminal departures "
              f"(StopSeq == 1)")
        print(f"    real mean   headway : {h_mean:7.1f} s")
        print(f"    real median headway : {h_median:7.1f} s")
        print(f"    real std    headway : {h_std:7.1f} s")
        print(f"  PLACEHOLDER (dataloader.dispatching_headway):")
        print(f"    hard-coded mean     : {PLACEHOLDER_HEADWAY_SEC:7.1f} s")
        print(f"    hard-coded std      : {PLACEHOLDER_HEADWAY_STD:7.1f} s")
        print(f"  --> the placeholder is ~{PLACEHOLDER_HEADWAY_SEC / h_mean:.2f}x "
              f"the real mean (i.e. roughly HALF the true headway).")
        print("  Fixing this is GAP 1 (Day 4). It also rescales every per-stop")
        print("  demand lambda, because lambda = boardings_per_trip / headway")
        print("  (that coupling is GAP 5 — see docs/03_data_to_env_mapping.md).")
    else:
        h_mean = h_median = h_std = float("nan")
        print("  WARNING: no terminal departures found — check StopSeq == 1.")

    # STUDENT TODO: try splitting `headways` by PeriodID (AM peak / Midday / PM
    # peak / Late evening). Day 4 formalises this "by_period" breakdown.

    # ----------------------------------------------------------------------- #
    # 6) AVL instantaneous speed (KPH) for the speed histogram.
    # ----------------------------------------------------------------------- #
    avl_dir = C.route_dir(avl_seg, direction)
    kph_all = pd.to_numeric(avl_dir["KPH"], errors="coerce").dropna().to_numpy()
    # Drop the corrupt high tail (see GAP-note: some KPH glitches exist).
    kph = kph_all[(kph_all >= 0.0) & (kph_all <= KPH_MAX_PLOT)]
    C.banner("AVL instantaneous speed (KPH)")
    print(f"  pings (this direction)   : {kph_all.size:,}")
    if kph.size:
        pct_zero = 100.0 * np.mean(kph == 0.0)
        print(f"  mean moving+dwell speed  : {np.mean(kph):.1f} km/h")
        print(f"  median speed             : {np.median(kph):.1f} km/h")
        print(f"  share of zero-speed pings: {pct_zero:.1f}%  (buses dwelling)")

    # ----------------------------------------------------------------------- #
    # 7) Make the four figures.
    # ----------------------------------------------------------------------- #
    C.banner("Saving figures")
    plot_boardings_per_stop(prof, direction)
    plot_load_profile(prof, direction)
    plot_headway_hist(headways, direction)
    plot_kph_hist(kph, direction)

    # ----------------------------------------------------------------------- #
    # 8) Save the headline numbers so later days / the write-up can cite them.
    # ----------------------------------------------------------------------- #
    summary = {
        "direction": direction,
        "generated_by": "day1_explore.py",
        "rows": {
            "apc_clean": int(apc_clean.shape[0]),
            "apc_raw": int(apc_raw.shape[0]),
            "avl_seg": int(avl_seg.shape[0]),
        },
        "date_min": str(date_min),
        "date_max": str(date_max),
        "branches": sorted(map(str, apc_clean["Branch"].dropna().unique())),
        "periods": sorted(map(str, apc_clean["PeriodID"].dropna().unique())),
        "daytypes": sorted(map(str, apc_clean["DAYTYPE"].dropna().unique())),
        "apc_latlon_coverage": prov["apc_latlon_coverage"],
        "gtfs_stop_overlap": prov["gtfs_stop_overlap"],
        "gtfs_calendar_start": prov["gtfs_calendar_start"],
        "gtfs_calendar_end": prov["gtfs_calendar_end"],
        "num_stops": int(n_stops),
        "terminal_start_id": str(term_start["StopID"]),
        "terminal_start_name": _short_stop_label(term_start),
        "terminal_end_id": str(term_end["StopID"]),
        "terminal_end_name": _short_stop_label(term_end),
        "unique_trips": n_trips,
        "total_boardings": total_boardings,
        "total_alightings": total_alightings,
        "mean_arrload": mean_arrload,
        "mean_depload": mean_depload,
        "mean_dwell_sec": mean_dwell,
        "headway_real_mean_sec": h_mean,
        "headway_real_median_sec": h_median,
        "headway_real_std_sec": h_std,
        "headway_placeholder_sec": PLACEHOLDER_HEADWAY_SEC,
        "headway_placeholder_std_sec": PLACEHOLDER_HEADWAY_STD,
        "avl_mean_kph": float(np.mean(kph)) if kph.size else float("nan"),
        "avl_zero_share": float(np.mean(kph == 0.0)) if kph.size else float("nan"),
    }
    C.save_json(summary, "day1_summary.json")

    C.banner("DAY 1 COMPLETE")
    print("  Look in student_project/outputs/ for four PNGs + day1_summary.json.")
    print("  Key takeaway: the shipped 300 s headway is ~half the real value.")
    print("  Next up: docs/04_week_plan.md (Day 2) and scripts/day2_geometry.py.")


if __name__ == "__main__":
    main()
