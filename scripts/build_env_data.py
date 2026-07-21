"""build_env_data.py — assemble the real-data TTC Route 29 environment.

This is the capstone script. It does NOT compute anything new itself: it calls
the four `derive_*` functions you wrote on Days 2-5 and stitches their outputs
into one `DataLoader`-compatible bundle per direction, then writes that bundle to
`student_project/outputs/derived/<north|south>/` in the SAME shapes the simulator
already knows how to read (see `setup/ttc_route_29_data/dataloader.py`).

    Day 2  geometry     -> node order + real inter-stop spacing (fixes GAP 2)
    Day 3  travel time  -> per-link tt_mean / tt_std / tt_cv     (link distributions)
    Day 4  headway      -> real dispatching headway (fixes GAP 1)
    Day 5  demand + OD   -> lambda per stop + IPF OD rate matrix (fixes GAP 5)

Run it standalone from the repo root:

    python student_project/scripts/build_env_data.py

It is NON-DESTRUCTIVE: it never writes under `setup/`. Everything lands in
`student_project/outputs/derived/`. At the end it prints a comparison against the
reference `setup/ttc_route_29_data/summary_<dir>.json` and a Day-6 pointer to
`day6_integrate.py`, which wires this regenerated data into the real simulator.

See docs/03_data_to_env_mapping.md for how each derived number maps onto a field
the `DataLoader` exposes, and docs/05_deliverables_checklist.md for "done".
"""

from __future__ import annotations

import sys
import json
import pickle
from pathlib import Path

# Mandatory header: make `common` and the sibling day scripts importable no
# matter what directory you launch from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C

# The Day 2-5 deliverables. Each exposes exactly one `derive_*` function that
# reads raw data and RETURNS values (no side effects). We only use those.
import day2_geometry
import day3_travel_time
import day4_headway
import day5_demand_od


# --------------------------------------------------------------------------- #
# Constants: the placeholders we are replacing (quote them so the improvement
# is visible in the printed report and in docs 03 / 04).
# --------------------------------------------------------------------------- #
OD_METHOD = "IPF (Iterative Proportional Fitting)"

# GAP 1: dataloader.py currently HARDCODES this (see its `dispatching_headway`).
PLACEHOLDER_HEADWAY = (300.0, 60.0)          # (mean_sec, std_sec)

# GAP 2: dataloader.py currently estimates spacing as tt_mean * 20 km/h
# (see its `get_spacing`). We replace it with real haversine distance.
PLACEHOLDER_SPEED_KMH = 20.0
PLACEHOLDER_SPEED_MPS = PLACEHOLDER_SPEED_KMH * 1000.0 / 3600.0

# Default link travel-time when a link has NO usable observations. This must
# match the fallback Day 3 uses so link_info stays self-consistent.
DEFAULT_TT_MEAN = 50.0
DEFAULT_TT_CV = 0.3
DEFAULT_TT_STD = DEFAULT_TT_MEAN * DEFAULT_TT_CV   # 15.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clean_str(x) -> str:
    """Return a stripped string, or '' for None/NaN."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x).strip()


def _stop_name(onstreet, atstreet) -> str:
    """Build a human stop name like 'Dufferin St at Wilson Ave'."""
    on, at = _clean_str(onstreet), _clean_str(atstreet)
    if on and at:
        return f"{on} at {at}"
    return on or at


# --------------------------------------------------------------------------- #
# The one assembly function the tests and main() both call.
# --------------------------------------------------------------------------- #
def derive_all(direction: str) -> dict:
    """Assemble every Day 2-5 result into one DataLoader-compatible dict.

    Parameter
    ---------
    direction : 'NORTH' or 'SOUTH' (case-insensitive).

    Returns
    -------
    A dict whose non-underscore keys mirror what
    `setup/ttc_route_29_data/dataloader.py` reads out of `data_<dir>.pickle`:

        node_ids              list[str], ordered terminal .. terminal
        terminal_start_id     str        (== node_ids[0])
        terminal_end_id       str        (== node_ids[-1])
        stop_pax_arrival_rate dict {stop_id: pax/sec}          (lambda, Day 5)
        link_time_info        dict {tail_stop_id: {loc, scale}} (Day 3)
        od_rate_table         dict {origin: {dest: pax/sec}}    (downstream only, Day 5)
        spacing               dict {tail_stop_id: meters}       (Day 2 — fixes GAP 2)
        dispatching_headway   (mean_sec, std_sec)               (Day 4 — fixes GAP 1)
        link_info             list[dict]  per-link tt + spacing
        stop_info             list[dict]  per-stop lambda + boardings
        direction, od_matrix_method       provenance

    Underscore-prefixed keys (_stops_df, _od_rate_df, ...) are convenience
    handles for main()'s CSV writers; they are stripped before pickling so the
    pickle stays lean and DataLoader-clean.
    """
    dir_up = direction.upper()
    if dir_up not in C.DIRECTIONS:
        raise ValueError(f"direction must be one of {C.DIRECTIONS}, got {direction!r}")

    # ---- Day 2: geometry (canonical node order + real spacing) -------------- #
    geo = day2_geometry.derive_geometry(dir_up)
    node_ids = [str(s) for s in geo["node_ids"]]
    spacing = {str(k): float(v) for k, v in geo["spacing"].items()}
    stops_df = geo["stops"].copy()
    # Look up a display name + cumulative distance per stop for the CSVs.
    name_by_id, cum_by_id = {}, {}
    for r in stops_df.itertuples(index=False):
        sid = str(getattr(r, "StopID"))
        name_by_id[sid] = _stop_name(getattr(r, "ONSTREET", ""),
                                     getattr(r, "ATSTREET", ""))
        cum_by_id[sid] = float(getattr(r, "cum_dist_m", np.nan))

    # ---- Day 3: link travel-time distributions ------------------------------ #
    tt = day3_travel_time.derive_travel_time(dir_up)
    # Index by (from, to) so we can walk the canonical node order below.
    tt_lookup = {
        (str(row.from_stop_id), str(row.to_stop_id)): row
        for row in tt.itertuples(index=False)
    }

    # ---- Day 4: real dispatching headway (fixes GAP 1) ---------------------- #
    hw = day4_headway.derive_headway(dir_up)
    hw_mean, hw_std = float(hw["overall"][0]), float(hw["overall"][1])

    # ---- Day 5: demand + OD (lambda is coupled to headway — GAP 5) ---------- #
    # STUDENT TODO: we pass the OVERALL headway so lambda_i = boardings/trip /
    # headway. If you later split headway by period (Day 4 `by_period`), you must
    # decide whether lambda should also be period-specific.
    dem = day5_demand_od.derive_demand_od(dir_up, headway_sec=hw_mean)
    lam = {str(k): float(v) for k, v in dem["lambda"].items()}
    boardings = {str(k): float(v) for k, v in dem["boardings"].items()}
    alightings = {str(k): float(v) for k, v in dem["alightings"].items()}
    od_df = dem["od_rate"]

    # ---- Assemble per-link structures in canonical order -------------------- #
    link_info, link_time_info = [], {}
    for seq, (frm, to) in enumerate(zip(node_ids[:-1], node_ids[1:])):
        row = tt_lookup.get((frm, to))
        if row is not None and int(getattr(row, "tt_count", 0)) > 0:
            tt_mean = float(row.tt_mean)
            tt_std = float(row.tt_std)
            tt_cv = float(row.tt_cv)
            tt_count = int(row.tt_count)
        else:
            # STUDENT TODO: no clean observations for this link — using a sane
            # default. Consider borrowing the neighbouring link, or the route
            # median, instead of a flat constant.
            tt_mean, tt_std, tt_cv, tt_count = (
                DEFAULT_TT_MEAN, DEFAULT_TT_STD, DEFAULT_TT_CV, 0)
        # Spacing is keyed by the DOWNSTREAM (tail) stop, matching both the
        # DataLoader convention and Day 2's geometry dict.
        spacing_m = float(spacing.get(to, np.nan))

        link_time_info[to] = {"loc": tt_mean, "scale": tt_std}
        link_info.append({
            "link_seq": seq,
            "from_stop_id": frm,
            "to_stop_id": to,
            "tt_mean": tt_mean,
            "tt_std": tt_std,
            "tt_cv": tt_cv,
            "tt_count": tt_count,
            "spacing_m": spacing_m,
        })

    # ---- Assemble per-stop structures --------------------------------------- #
    stop_info = []
    for i, sid in enumerate(node_ids):
        lam_i = lam.get(sid, 0.0)
        stop_info.append({
            "StopID": sid,
            "stop_seq": i + 1,
            "lambda": lam_i,             # pax/sec  (contract key)
            "boardings": boardings.get(sid, 0.0),
            # extras below are handy for the stops CSV; DataLoader ignores them.
            "alightings": alightings.get(sid, 0.0),
            "lambda_pax_min": lam_i * 60.0,
            "stop_name": name_by_id.get(sid, sid),
            "cum_dist_m": cum_by_id.get(sid, np.nan),
        })

    # ---- OD rate table: nested dict, downstream (upper-triangular) only ------ #
    od_rate_table: dict = {}
    for origin in od_df.index:
        dests = {}
        for dest, val in od_df.loc[origin].items():
            v = float(val)
            if v > 0.0:                  # IPF seeds only downstream pairs
                dests[str(dest)] = v
        if dests:
            od_rate_table[str(origin)] = dests

    result = {
        # --- DataLoader-compatible payload (goes into the pickle) ---
        "direction": dir_up,
        "node_ids": node_ids,
        "terminal_start_id": node_ids[0],
        "terminal_end_id": node_ids[-1],
        "stop_pax_arrival_rate": lam,
        "link_time_info": link_time_info,
        "stop_info": stop_info,
        "link_info": link_info,
        "od_rate_table": od_rate_table,
        "spacing": spacing,                      # fixes GAP 2 (real haversine)
        "dispatching_headway": (hw_mean, hw_std),  # fixes GAP 1 (real headway)
        "od_matrix_method": OD_METHOD,
        # --- convenience handles for main() only (stripped before pickling) ---
        "_stops_df": stops_df,
        "_od_rate_df": od_df,
        "_travel_time_df": tt,
        "_headway_by_period": hw.get("by_period", {}),
    }
    return result


# --------------------------------------------------------------------------- #
# Output builders (mirror the reference format in setup/ttc_route_29_data/)
# --------------------------------------------------------------------------- #
def build_summary(data: dict) -> dict:
    """A summary_<dir>.json in the same shape as the reference file."""
    link_info = data["link_info"]
    with_data = [l for l in link_info if l["tt_count"] > 0]
    mean_tt = float(np.mean([l["tt_mean"] for l in with_data])) if with_data else 0.0
    mean_cv = float(np.mean([l["tt_cv"] for l in with_data])) if with_data else 0.0

    lam_vals = list(data["stop_pax_arrival_rate"].values())
    mean_lam_min = float(np.mean(lam_vals) * 60.0) if lam_vals else 0.0
    total_b = float(sum(s["boardings"] for s in data["stop_info"]))

    hw_mean, hw_std = data["dispatching_headway"]
    return {
        "direction": data["direction"],
        "num_stops": len(data["node_ids"]),
        "terminal_start_id": data["terminal_start_id"],
        "terminal_end_id": data["terminal_end_id"],
        "terminal_start_name": data["stop_info"][0]["stop_name"],
        "terminal_end_name": data["stop_info"][-1]["stop_name"],
        "total_boardings": total_b,
        "mean_lambda_pax_min": mean_lam_min,
        "mean_tt_sec": mean_tt,
        "mean_tt_cv": mean_cv,
        "num_links_with_data": len(with_data),
        # New fields that make the fixed placeholders explicit:
        "dispatching_headway_mean_sec": float(hw_mean),
        "dispatching_headway_std_sec": float(hw_std),
        "headway_placeholder_sec": PLACEHOLDER_HEADWAY[0],
        "od_matrix_method": data["od_matrix_method"],
    }


def build_stops_csv(data: dict) -> pd.DataFrame:
    """A stops_<dir>.csv mirroring the reference columns we can fill."""
    rows = []
    for s in data["stop_info"]:
        rows.append({
            "stop_id": s["StopID"],
            "stop_seq": s["stop_seq"],
            "stop_name": s["stop_name"],
            "total_boardings": s["boardings"],
            "total_alightings": s["alightings"],
            "lambda_pax_sec": s["lambda"],
            "lambda_pax_min": s["lambda_pax_min"],
            "cum_dist_m": s["cum_dist_m"],
        })
    return pd.DataFrame(rows)


def build_links_csv(data: dict) -> pd.DataFrame:
    """A links_<dir>.csv: reference columns plus the real spacing_m (GAP 2)."""
    cols = ["link_seq", "from_stop_id", "to_stop_id",
            "tt_mean", "tt_std", "tt_cv", "tt_count", "spacing_m"]
    return pd.DataFrame(data["link_info"])[cols]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_comparison(direction: str, data: dict, summary: dict) -> None:
    """Print regenerated-vs-reference numbers so the student sees the deltas."""
    dl = direction.lower()
    ref_path = C.SETUP_DATA / f"summary_{dl}.json"
    ref = {}
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)

    C.banner(f"COMPARISON  ({direction})  reference/placeholder  vs  regenerated")
    print(f"  {'metric':<26}{'reference / OLD':>20}{'regenerated / NEW':>20}")
    print("  " + "-" * 64)

    def row(metric, old, new, fmt="{:.2f}"):
        old_s = fmt.format(old) if isinstance(old, (int, float)) else str(old)
        new_s = fmt.format(new) if isinstance(new, (int, float)) else str(new)
        print(f"  {metric:<26}{old_s:>20}{new_s:>20}")

    row("num_stops", ref.get("num_stops", "n/a"), summary["num_stops"], "{:d}")
    row("mean_tt_sec", ref.get("mean_tt_sec", float("nan")), summary["mean_tt_sec"])
    row("mean_tt_cv", ref.get("mean_tt_cv", float("nan")), summary["mean_tt_cv"], "{:.3f}")

    # GAP 1: headway. The OLD value is the hardcoded placeholder in dataloader.py.
    hw_mean, hw_std = data["dispatching_headway"]
    row("headway_mean_sec", PLACEHOLDER_HEADWAY[0], hw_mean)
    row("headway_std_sec", PLACEHOLDER_HEADWAY[1], hw_std)

    # GAP 2: spacing. OLD = tt_mean * 20 km/h; NEW = real haversine distance.
    with_sp = [l for l in data["link_info"] if not np.isnan(l["spacing_m"])]
    if with_sp:
        old_mean_sp = float(np.mean([l["tt_mean"] * PLACEHOLDER_SPEED_MPS
                                     for l in with_sp]))
        new_mean_sp = float(np.mean([l["spacing_m"] for l in with_sp]))
        row("mean_spacing_m", old_mean_sp, new_mean_sp)

    # GAP 5: lambda is coupled to headway. Because the real headway (~500-550s)
    # is roughly double the 300s placeholder, lambda roughly halves.
    print(f"\n  NOTE (GAP 5): lambda_i = boardings_per_trip / headway. Headway went "
          f"{PLACEHOLDER_HEADWAY[0]:.0f}s -> {hw_mean:.0f}s,")
    print(f"               so per-stop arrival rates scale by "
          f"{PLACEHOLDER_HEADWAY[0] / hw_mean:.2f}x versus the old placeholder.")


def print_graduation_steps() -> None:
    """How to later wire this regenerated data into the real simulator.

    We deliberately DO NOT perform these edits — they touch setup/, which this
    script must never modify. This is the student's final, reviewed hand-off.
    """
    bar = "#" * 72
    print("\n" + bar)
    print("# NEXT: DAY 6 — wire this regenerated data into the real simulator")
    print(bar)
    print(f"""
  Everything above was written to student_project/outputs/derived/<north|south>/
  and NOTHING under setup/ was touched.

  Day 6 is YOUR open-ended capstone: wire this bundle into the simulator so the TTC
  env sources real values for the two placeholders, then run it and measure bunching:
      GAP 1 headway : (300, 60)             -> real (mean, std) from this bundle
      GAP 2 spacing : tt_mean * 20 km/h     -> real haversine spacing from this bundle

  The tooling helps, it does NOT do the wiring for you:
      python scripts/day6_integrate.py           # preview the target numbers (safe)
      python scripts/day6_integrate.py --check    # validate the env YOU build
      python quick_run.py --plot                  # run it; then compare a holding agent

  Full guide (contract + design options): docs/06_integrate_into_simulator.md
  (Re-running THIS script rebuilds every derived file from raw_data/ — that is
  GAP 3 closed: the pipeline is the single reproducible source of truth.)
""")
    print(bar)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    C.banner("build_env_data.py — assemble the real-data TTC Route 29 environment")
    print("Reads Day 2-5 derivations, writes a DataLoader-compatible bundle per")
    print("direction to student_project/outputs/derived/. Nothing under setup/ is")
    print("touched. Placeholders being replaced: headway (300,60) and 20 km/h spacing.")

    for direction in C.DIRECTIONS:
        dl = direction.lower()
        C.banner(f"DIRECTION: {direction}")

        data = derive_all(direction)

        # 1) the pickle — strip the private '_' handles so it stays DataLoader-clean.
        pickle_dict = {k: v for k, v in data.items() if not k.startswith("_")}
        pkl_path = C.out_path(f"derived/{dl}/data_{dl}.pickle")
        with open(pkl_path, "wb") as f:
            pickle.dump(pickle_dict, f)
        print(f"  [saved] {pkl_path.relative_to(C.REPO_ROOT)}  "
              f"({len(pickle_dict)} keys)")

        # 2) summary json
        summary = build_summary(data)
        C.save_json(summary, f"derived/{dl}/summary_{dl}.json")

        # 3) stops csv
        C.save_csv(build_stops_csv(data), f"derived/{dl}/stops_{dl}.csv")

        # 4) links csv
        C.save_csv(build_links_csv(data), f"derived/{dl}/links_{dl}.csv")

        # 5) OD rate matrix csv (origin x dest, upper-triangular pax/sec).
        # Blank the index name to match the reference's empty top-left cell.
        od_df = data["_od_rate_df"].copy()
        od_df.index.name = None
        C.save_csv(od_df, f"derived/{dl}/od_rate_matrix_{dl}.csv", index=True)

        # 6) show the improvement over the placeholders / reference.
        print_comparison(direction, data, summary)

    print_graduation_steps()


if __name__ == "__main__":
    main()
