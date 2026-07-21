"""
day3_travel_time.py — derive per-link (stop-to-stop) travel-time distributions
for TTC Route 29 from the real APC data, replacing the simulator's guessed link
travel times with numbers measured from the November-2023 feed.

WHAT THIS SCRIPT PRODUCES
-------------------------
For one direction ("NORTH" or "SOUTH") we build a table with one row per
*canonical link* (a consecutive pair of stops along the route):

    from_stop_id, to_stop_id, tt_mean, tt_std, tt_cv, tt_count

`tt_*` are in SECONDS. This is exactly the shape `build_env_data.py` needs to
fill the DataLoader's `link_time_info` ({tail_stop_id: {loc: mean, scale: std}}).

HOW WE MEASURE A LINK TRAVEL TIME
---------------------------------
Every APC row is one bus stopping at one stop, with a StopArrivalTime and a
StopDepartureTime. For a single bus run (one TripID on one service date) the
time it spent *travelling* from stop A to the next stop B is:

    travel_time(A -> B) = arrival_at_B  -  departure_from_A     (seconds)

i.e. next stop's arrival minus this stop's departure. (Subtracting the departure,
not the arrival, means we exclude the dwell time spent at A — that is boarding,
not travelling.) We collect this over every trip and every day, then take the
mean / std per link.

WHY THIS MATTERS (the gap we are closing)
-----------------------------------------
`setup/ttc_route_29_data/dataloader.py` already stores real link times, but the
travel time is *also* what the old placeholder geometry leaned on: `get_spacing()`
multiplies `tt_mean` by a flat 20 km/h to invent distances (see day2_geometry.py,
GAP 2). Getting `tt_mean` right and honest is therefore load-bearing for both the
travel-time model AND the fallback geometry. See docs/03_data_to_env_mapping.md.

RUN IT
------
    cd /home/jiahao/Documents/busoperation
    python student_project/scripts/day3_travel_time.py            # both directions
    python student_project/scripts/day3_travel_time.py NORTH      # just one

`derive_travel_time(direction)` is importable and has no side effects (it only
reads data and returns a DataFrame) — build_env_data.py and the tests call it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (path set up above, must come first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE pyplot
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Tunable constants — the honest "decision knobs" of this script.
# --------------------------------------------------------------------------- #
# Keep only physically plausible stop-to-stop times. Under 5 s is a GPS/APC
# glitch or two records of the same stop; over 1200 s (20 min) between two
# ADJACENT stops means the trip was interrupted / stitched wrong.
TT_MIN_SEC = 5.0
TT_MAX_SEC = 1200.0

# Fallback for a link that never got a clean observation (e.g. a branch-only
# stop pair). Stated openly so nobody mistakes it for measured data.
DEFAULT_TT_MEAN = 50.0
DEFAULT_TT_CV = 0.3
DEFAULT_TT_STD = DEFAULT_TT_MEAN * DEFAULT_TT_CV  # 15.0 s

# The number the reference environment reports, for the sanity cross-check.
# (setup/ttc_route_29_data/summary_north.json -> mean_tt_sec)
REFERENCE_MEAN_TT_SEC = 50.0


# --------------------------------------------------------------------------- #
# Core derivation  (pure: reads data, returns a DataFrame, writes nothing)
# --------------------------------------------------------------------------- #
def derive_travel_time(direction: str) -> pd.DataFrame:
    """Measure stop-to-stop travel time per canonical link for one direction.

    Args:
        direction: "NORTH" or "SOUTH" (case-insensitive).

    Returns:
        DataFrame with one row per consecutive stop pair along the canonical
        route order, columns:
            from_stop_id (str), to_stop_id (str),
            tt_mean (s), tt_std (s), tt_cv, tt_count (int)
        Links with no clean observation are filled with the stated defaults
        (tt_mean=50, tt_cv=0.3, tt_count=0).
    """
    apc = C.load_apc_clean()

    # 1) Canonical stop order (terminal .. terminal) shared by every day script.
    stops = C.stop_sequence(apc, direction)
    ordered_ids = stops["StopID"].astype(str).tolist()
    # position of each stop in the canonical order: {stop_id -> 0,1,2,...}
    order = {sid: i for i, sid in enumerate(ordered_ids)}

    # 2) Filter to this route+direction and parse the timestamps.
    d = C.route_dir(apc, direction).copy()
    d["StopID"] = d["StopID"].astype(str)
    d["arr"] = pd.to_datetime(d["StopArrivalTime"], errors="coerce")
    d["dep"] = pd.to_datetime(d["StopDepartureTime"], errors="coerce")
    d["ord"] = d["StopID"].map(order)
    # A row is only usable if we could parse both times and place the stop.
    d = d.dropna(subset=["arr", "dep", "ord"])
    d["ord"] = d["ord"].astype(int)

    # 3) A "bus run" is one TripID on one service date. TripIDs (e.g.
    #    "9151_N_0_DLWI") repeat across days, so grouping by TripID alone would
    #    wrongly chain the last stop of one day to the first stop of the next.
    #    Adding the calendar date keeps each run intact.
    d["service_date"] = d["arr"].dt.normalize()

    # Order each run by the stop sequence the bus actually visited, then look at
    # the *next* stop in that run.
    d = d.sort_values(["TripID", "service_date", "StopSeq"])
    run = d.groupby(["TripID", "service_date"], sort=False)
    d["next_ord"] = run["ord"].shift(-1)
    d["next_id"] = run["StopID"].shift(-1)
    d["next_arr"] = run["arr"].shift(-1)

    # 4) Keep only pairs that are ADJACENT in the canonical order (next_ord ==
    #    ord + 1). This throws away hops that skip a stop (branch trips) or that
    #    jump across a day boundary, so every measurement lands on a real link.
    adj = d[d["next_ord"] == d["ord"] + 1].copy()
    adj["tt"] = (adj["next_arr"] - adj["dep"]).dt.total_seconds()

    # 5) Drop physically impossible durations before aggregating.
    #    STUDENT TODO: these bounds are deliberately loose. Tighten them, or
    #    switch to a per-link percentile clip (e.g. keep the 1st..99th pct), if
    #    you find residual outliers skewing a particular link's mean.
    adj = adj[(adj["tt"] >= TT_MIN_SEC) & (adj["tt"] <= TT_MAX_SEC)]

    # STUDENT TODO (time-of-day): PeriodID (AM peak / Midday / PM peak / ...)
    # is available on every row. Route 29 is much slower in the PM peak than at
    # night. To model that, groupby ["StopID", "next_id", "PeriodID"] here and
    # return period-specific distributions, then let the simulator pick the
    # active period. The default below collapses all periods into one number.

    measured = (
        adj.groupby(["StopID", "next_id"])["tt"]
        .agg(tt_mean="mean", tt_std="std", tt_count="count")
        .reset_index()
        .rename(columns={"StopID": "from_stop_id", "next_id": "to_stop_id"})
    )
    # std is NaN when a link has exactly one observation — treat as 0 spread.
    measured["tt_std"] = measured["tt_std"].fillna(0.0)

    # 6) Build the full canonical link table (every consecutive pair), then
    #    attach the measurements. Links with no data get the stated defaults.
    links = pd.DataFrame(
        {
            "from_stop_id": ordered_ids[:-1],
            "to_stop_id": ordered_ids[1:],
        }
    )
    out = links.merge(measured, on=["from_stop_id", "to_stop_id"], how="left")

    no_data = out["tt_count"].isna()
    out.loc[no_data, "tt_mean"] = DEFAULT_TT_MEAN
    out.loc[no_data, "tt_std"] = DEFAULT_TT_STD
    out["tt_count"] = out["tt_count"].fillna(0).astype(int)

    # coefficient of variation; guard the (impossible) zero-mean case.
    out["tt_cv"] = np.where(out["tt_mean"] > 0, out["tt_std"] / out["tt_mean"], 0.0)
    # Give the fallback links their stated cv exactly (mean/std were defaults).
    out.loc[no_data, "tt_cv"] = DEFAULT_TT_CV

    # STUDENT TODO (distribution shape): we only keep mean+std, which the
    # simulator uses as a normal/lognormal loc+scale. Travel times are usually
    # right-skewed (a long tail of delayed trips). If you want a better fit,
    # test lognormal or gamma per link (scipy.stats.lognorm.fit) and hand the
    # extra shape parameter through link_time_info.

    out = out[["from_stop_id", "to_stop_id", "tt_mean", "tt_std", "tt_cv", "tt_count"]]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# AVL speed cross-check  (independent sanity check, not used by build_env_data)
# --------------------------------------------------------------------------- #
def avl_speed_crosscheck(direction: str) -> dict:
    """Summarise instantaneous GPS speed (KPH) from the segmented AVL feed.

    The APC-derived travel times and the AVL speeds come from two different
    sensors; if they roughly agree we trust both. Many KPH readings are 0
    because the bus is dwelling at a stop or in traffic, so we report the mean
    both with and without the zeros.
    """
    avl = C.load_avl_seg()
    a = C.route_dir(avl, direction)
    kph = pd.to_numeric(a["KPH"], errors="coerce").dropna()
    # Guard against corrupt highway-speed spikes noted in the data dictionary.
    kph = kph[(kph >= 0) & (kph <= 100)]
    zero_frac = float((kph == 0).mean()) if len(kph) else float("nan")
    moving = kph[kph > 0]
    return {
        "n_pings": int(len(kph)),
        "mean_kph_all": float(kph.mean()) if len(kph) else float("nan"),
        "mean_kph_moving": float(moving.mean()) if len(moving) else float("nan"),
        "zero_fraction": zero_frac,
    }


# --------------------------------------------------------------------------- #
# Reporting helpers (only used by main())
# --------------------------------------------------------------------------- #
def _load_reference_mean_tt(direction: str):
    """Read mean_tt_sec from the reference summary_<dir>.json if it exists."""
    import json

    path = C.SETUP_DATA / f"summary_{direction.lower()}.json"
    if not path.exists():
        return None
    with open(path) as f:
        summary = json.load(f)
    return summary.get("mean_tt_sec")


def _plot_travel_time(tt: pd.DataFrame, direction: str) -> None:
    """Line of tt_mean along the route with a shaded +/- std band."""
    x = np.arange(len(tt))
    mean = tt["tt_mean"].to_numpy()
    std = tt["tt_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color="tab:blue",
                    label="+/- 1 std")
    ax.plot(x, mean, color="tab:blue", marker="o", markersize=3, linewidth=1.2,
            label="tt_mean")
    # Mark links that fell back to the default so the student can see coverage.
    nodata = tt["tt_count"] == 0
    if nodata.any():
        ax.scatter(x[nodata.to_numpy()], mean[nodata.to_numpy()], color="crimson",
                   zorder=5, s=28, label="no data (default)")
    ax.set_xlabel("link index (terminal -> terminal)")
    ax.set_ylabel("stop-to-stop travel time (s)")
    ax.set_title(f"TTC Route 29 {direction.upper()} — measured link travel time")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    C.savefig(fig, f"day3_travel_time_{direction.lower()}.png")
    plt.close(fig)


def run_direction(direction: str) -> pd.DataFrame:
    """Derive, report, and save travel times for one direction. Returns table."""
    C.banner(f"DAY 3 — TRAVEL TIME  ({direction.upper()})")

    tt = derive_travel_time(direction)

    n_links = len(tt)
    n_data = int((tt["tt_count"] > 0).sum())
    mean_tt = float(tt.loc[tt["tt_count"] > 0, "tt_mean"].mean())
    mean_cv = float(tt.loc[tt["tt_count"] > 0, "tt_cv"].mean())
    total_obs = int(tt["tt_count"].sum())

    print(f"\nCanonical links           : {n_links}")
    print(f"Links with measured data  : {n_data}  "
          f"(fell back to default: {n_links - n_data})")
    print(f"Total stop-to-stop samples: {total_obs:,}")
    print(f"Mean link travel time     : {mean_tt:7.2f} s")
    print(f"Mean link cv (std/mean)   : {mean_cv:7.3f}")

    print("\nSegment table (first 8 links):")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(tt.head(8).to_string(index=False))

    # ---- cross-check against the reference environment -------------------- #
    ref_mean = _load_reference_mean_tt(direction)
    print("\nCross-check vs reference environment:")
    if ref_mean is not None:
        delta = mean_tt - ref_mean
        print(f"  reference mean_tt_sec (summary_{direction.lower()}.json) "
              f": {ref_mean:7.2f} s")
        print(f"  our mean link travel time                    : {mean_tt:7.2f} s")
        print(f"  difference                                   : {delta:+7.2f} s")
        if abs(delta) <= 5.0:
            print("  -> within 5 s: our pipeline reproduces the reference. Good.")
        else:
            print("  -> differs by >5 s: check the outlier bounds / stop order.")
    else:
        print(f"  (no reference summary_{direction.lower()}.json found — expected "
              f"~{REFERENCE_MEAN_TT_SEC:.0f} s for NORTH)")

    # ---- independent AVL speed cross-check -------------------------------- #
    spd = avl_speed_crosscheck(direction)
    print("\nAVL GPS speed cross-check (independent sensor):")
    print(f"  usable pings              : {spd['n_pings']:,}")
    print(f"  mean speed (incl. zeros)  : {spd['mean_kph_all']:6.2f} km/h")
    print(f"  mean speed (moving only)  : {spd['mean_kph_moving']:6.2f} km/h")
    print(f"  fraction of zero-speed    : {spd['zero_fraction']:6.1%}  "
          f"(dwelling / stopped in traffic)")
    print("  NOTE: the old geometry placeholder assumed a flat 20 km/h for the "
          "WHOLE\n        route (day2 GAP 2); the moving speed above shows how "
          "rough that is.")

    # ---- save outputs ----------------------------------------------------- #
    print()
    C.save_csv(tt, f"day3_travel_time_{direction.lower()}.csv")
    _plot_travel_time(tt, direction)

    return tt


def main() -> None:
    # Optional CLI arg to run a single direction, else run both.
    args = [a.upper() for a in sys.argv[1:] if a.upper() in C.DIRECTIONS]
    directions = args if args else list(C.DIRECTIONS)
    for direction in directions:
        run_direction(direction)

    C.banner("DAY 3 DONE")
    print("Outputs are in student_project/outputs/ (day3_travel_time_*.csv / .png).")
    print("Next: day4_headway.py fixes the (300, 60) headway placeholder (GAP 1),")
    print("then build_env_data.py folds these link times into link_time_info.")


if __name__ == "__main__":
    main()
