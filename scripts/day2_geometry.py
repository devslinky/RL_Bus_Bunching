"""
day2_geometry.py — derive REAL stop geometry (spacing) for TTC Route 29.

WHAT THIS FIXES (GAP 2)
-----------------------
The live simulator loads spacing from
    setup/ttc_route_29_data/dataloader.py :: DataLoader.get_spacing()
which does NOT use any real distance. It *guesses*:

    avg_speed_mps = 20 * 1000 / 3600          # 20 km/h, an urban-bus rule of thumb
    spacing[to_stop] = tt_mean * avg_speed_mps # meters = seconds * (m/s)

i.e. every inter-stop distance is invented from a travel-time mean times a flat
20 km/h. That is fine for a toy, but we can do far better: the raw APC table
`ttc_apc_data.csv` carries a per-stop Lat/Lon with ~100% coverage on Route 29,
so the TRUE inter-stop spacing is just the cumulative great-circle (haversine)
distance walked along the canonical stop order.

This script computes that real spacing, saves it, draws a route map, and prints
a side-by-side comparison of the real distances against the 20-km/h placeholder.

HOW TO RUN (from the repo root /home/jiahao/Documents/busoperation)
-------------------------------------------------------------------
    python student_project/scripts/day2_geometry.py            # both directions
    python student_project/scripts/day2_geometry.py NORTH      # one direction

Outputs land in student_project/outputs/ (never in setup/ or raw_data/).

WHAT "DONE" LOOKS LIKE
----------------------
* A printed stop table with a growing `cum_dist_m` column and a sane total
  route length (Route 29 DUFFERIN is roughly 15-17 km end to end).
* outputs/day2_geometry_<dir>.csv  and  outputs/day2_route_map_<dir>.png exist.
* `from day2_geometry import derive_geometry` works and returns a dict whose
  "spacing" is consumed by build_env_data.py.

See also:
    docs/03_data_to_env_mapping.md   (GAP 2, and how spacing feeds the env)
    docs/04_week_plan.md             (Day 2)
    student_project/scripts/day3_travel_time.py  (the tt this spacing replaces)
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- make the sibling helper module importable, then import it (mandatory) --- #
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # headless: pick the non-interactive backend BEFORE pyplot
import matplotlib.pyplot as plt  # noqa: E402


# The placeholder "speed" baked into DataLoader.get_spacing(); kept here only so
# main() can reproduce the old estimate for the comparison print-out.
PLACEHOLDER_SPEED_KMH = 20.0
PLACEHOLDER_SPEED_MPS = PLACEHOLDER_SPEED_KMH * 1000.0 / 3600.0  # ~5.556 m/s


# --------------------------------------------------------------------------- #
# Core deriver  (importable, NO side effects — only reads data and returns)
# --------------------------------------------------------------------------- #
def derive_geometry(direction: str) -> dict:
    """Derive real stop geometry for one direction of TTC Route 29.

    Parameters
    ----------
    direction : str
        "NORTH" or "SOUTH" (either case accepted).

    Returns
    -------
    dict with keys
        "node_ids" : list[str]
            StopIDs ordered terminal..terminal, exactly the C.stop_sequence order.
        "spacing" : dict[str, float]
            {downstream_stop_id: meters} for every consecutive link. Keyed by the
            DOWNSTREAM (tail) stop to match the DataLoader convention, so the very
            first (terminal) stop has NO entry. Values are haversine distances.
        "stops" : pandas.DataFrame
            columns [StopID, stop_seq, ONSTREET, ATSTREET, Lat, Lon, cum_dist_m],
            ordered terminal..terminal. cum_dist_m[0] == 0.0.
    """
    direction = direction.upper()

    # 1) Canonical ordered stop list (terminal..terminal). C.stop_sequence already
    #    de-duplicates physical stops and orders them by median StopSeq, which is
    #    robust to occasional mis-sequenced trips and to branch differences.
    apc_clean = C.load_apc_clean()
    seq = C.stop_sequence(apc_clean, direction)  # StopID(str), stop_seq, ONSTREET, ATSTREET, n_trips

    # 2) Per-stop position. We deliberately use the raw APC Lat/Lon (median over
    #    all trips) rather than the 2020 GTFS stops.txt: the GTFS feed is a 2020
    #    snapshot and only ~72% of 2023 stop_ids appear in it (GAP 4). The APC
    #    Lat/Lon is contemporaneous with the AVL/APC we model and ~100% covered.
    apc_raw = C.load_apc_raw()
    latlon = C.stop_latlon(apc_raw, direction)  # StopID(str), Lat, Lon

    # 3) Attach positions to the ordered sequence (left join keeps the order).
    stops = seq.merge(latlon, on="StopID", how="left")
    stops = stops.sort_values("stop_seq").reset_index(drop=True)
    stops["StopID"] = stops["StopID"].astype(str)

    # Guard: with ~100% coverage this should never fire, but be robust so we
    # never emit NaN spacing downstream. Any stop missing a position gets one
    # linearly interpolated from its neighbours (order-preserving, monotonic).
    n_missing = int(stops[["Lat", "Lon"]].isna().any(axis=1).sum())
    if n_missing:
        stops["Lat"] = stops["Lat"].interpolate(limit_direction="both")
        stops["Lon"] = stops["Lon"].interpolate(limit_direction="both")

    # 4) Per-link haversine distance and cumulative distance along the route.
    lat = stops["Lat"].to_numpy()
    lon = stops["Lon"].to_numpy()
    link_dist = np.zeros(len(stops), dtype=float)  # index i = distance from i-1 -> i
    for i in range(1, len(stops)):
        link_dist[i] = C.haversine(lat[i - 1], lon[i - 1], lat[i], lon[i])

    stops["cum_dist_m"] = np.cumsum(link_dist)  # cum_dist_m[0] == 0.0 by construction

    # 5) Spacing dict keyed by the DOWNSTREAM stop (matches DataLoader.get_spacing).
    #    The first stop (a terminal) has no upstream neighbour, hence no entry.
    spacing = {
        str(stops.loc[i, "StopID"]): float(link_dist[i])
        for i in range(1, len(stops))
    }

    return {
        "node_ids": stops["StopID"].tolist(),
        "spacing": spacing,
        "stops": stops[
            ["StopID", "stop_seq", "ONSTREET", "ATSTREET", "Lat", "Lon", "cum_dist_m"]
        ].copy(),
        # extra bookkeeping (harmless to consumers that only read the 3 keys above)
        "n_missing_latlon": n_missing,
    }


# --------------------------------------------------------------------------- #
# Comparison against the old placeholder  (reporting only — no side effects on
# anything except stdout / the returned frame)
# --------------------------------------------------------------------------- #
def _placeholder_spacing(direction: str) -> pd.DataFrame | None:
    """Reproduce the OLD 20-km/h spacing from the reference links CSV, if present.

    The old formula (DataLoader.get_spacing) is:
        spacing[to_stop] = tt_mean * (20 km/h in m/s)
    We read the reference tt_mean per link from setup/ttc_route_29_data/links_<dir>.csv
    so the student can see, link by link, how far off the guess was.

    Returns a DataFrame [to_stop_id(str), placeholder_m] or None if the reference
    file is missing (e.g. a fresh checkout without the reference derived data).
    """
    ref = C.SETUP_DATA / f"links_{direction.lower()}.csv"
    if not ref.exists():
        return None
    links = pd.read_csv(ref)
    links["to_stop_id"] = links["to_stop_id"].astype(str)
    links["placeholder_m"] = links["tt_mean"].astype(float) * PLACEHOLDER_SPEED_MPS
    return links[["to_stop_id", "placeholder_m"]]


def _build_report_frame(geo: dict, direction: str) -> pd.DataFrame:
    """Join real spacing with the placeholder spacing for the print-out / CSV."""
    stops = geo["stops"].copy()
    stops["spacing_real_m"] = stops["StopID"].map(geo["spacing"])  # NaN at terminal row

    placeholder = _placeholder_spacing(direction)
    if placeholder is not None:
        stops = stops.merge(
            placeholder, left_on="StopID", right_on="to_stop_id", how="left"
        ).drop(columns="to_stop_id")
    else:
        stops["placeholder_m"] = np.nan
    return stops


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _plot_route_map(stops: pd.DataFrame, direction: str):
    """Lon-vs-Lat map: the stop polyline with both terminals highlighted."""
    fig, ax = plt.subplots(figsize=(6, 9))

    # The ordered polyline of stops (this is the physical shape of the route).
    ax.plot(stops["Lon"], stops["Lat"], "-", color="#4c78a8", lw=1.5, zorder=1,
            label="stop-to-stop path")
    ax.scatter(stops["Lon"], stops["Lat"], s=18, color="#4c78a8", zorder=2)

    # Highlight terminals (first and last rows in the canonical order).
    term = stops.iloc[[0, -1]]
    ax.scatter(term["Lon"], term["Lat"], s=140, marker="*", color="#e45756",
               edgecolor="black", zorder=3, label="terminals")
    for _, r in term.iterrows():
        ax.annotate(f"{r['StopID']}\n{r['ONSTREET']} @ {r['ATSTREET']}",
                    (r["Lon"], r["Lat"]), fontsize=7,
                    xytext=(6, 0), textcoords="offset points", va="center")

    total_km = stops["cum_dist_m"].iloc[-1] / 1000.0
    ax.set_title(f"TTC Route 29 {direction} — {len(stops)} stops, "
                 f"{total_km:.1f} km (real geometry)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")  # keep the map from looking squashed
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


# --------------------------------------------------------------------------- #
# Standalone entry point
# --------------------------------------------------------------------------- #
def _run_one(direction: str) -> None:
    direction = direction.upper()
    C.banner(f"DAY 2 — GEOMETRY / SPACING   ({direction})")

    geo = derive_geometry(direction)
    stops = _build_report_frame(geo, direction)

    if geo["n_missing_latlon"]:
        print(f"  NOTE: {geo['n_missing_latlon']} stop(s) had no APC Lat/Lon; "
              f"positions were interpolated from neighbours.")

    # --- stop table with cumulative distance -------------------------------- #
    show = stops[["stop_seq", "StopID", "ONSTREET", "ATSTREET",
                  "spacing_real_m", "cum_dist_m"]].copy()
    show["spacing_real_m"] = show["spacing_real_m"].round(1)
    show["cum_dist_m"] = show["cum_dist_m"].round(1)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(show.to_string(index=False))

    total_km = stops["cum_dist_m"].iloc[-1] / 1000.0
    n_links = len(geo["spacing"])
    mean_spacing = np.mean(list(geo["spacing"].values())) if n_links else float("nan")
    print(f"\n  Stops (incl. terminals) : {len(stops)}")
    print(f"  Links                   : {n_links}")
    print(f"  Total route length      : {total_km:.2f} km")
    print(f"  Mean inter-stop spacing : {mean_spacing:.1f} m")

    # --- comparison: REAL geometry vs the 20 km/h placeholder --------------- #
    C.banner(f"REAL SPACING  vs  PLACEHOLDER (tt_mean * {PLACEHOLDER_SPEED_KMH:.0f} km/h)")
    if stops["placeholder_m"].notna().any():
        cmp = stops.dropna(subset=["spacing_real_m", "placeholder_m"]).copy()
        cmp["abs_err_m"] = (cmp["spacing_real_m"] - cmp["placeholder_m"]).abs()
        print("  Placeholder formula (setup/.../dataloader.py::get_spacing):")
        print(f"      spacing[to_stop] = tt_mean * ({PLACEHOLDER_SPEED_KMH:.0f} km/h "
              f"= {PLACEHOLDER_SPEED_MPS:.3f} m/s)\n")
        head = cmp[["stop_seq", "StopID", "spacing_real_m", "placeholder_m", "abs_err_m"]].head(12)
        with pd.option_context("display.width", 200):
            print(head.round(1).to_string(index=False))
        real_total = cmp["spacing_real_m"].sum() / 1000.0
        place_total = cmp["placeholder_m"].sum() / 1000.0
        print(f"\n  Route length REAL        : {real_total:6.2f} km")
        print(f"  Route length PLACEHOLDER : {place_total:6.2f} km  "
              f"(20 km/h guess)")
        print(f"  Mean absolute per-link error : {cmp['abs_err_m'].mean():.1f} m")
        print("  --> real geometry replaces a speed guess with measured distance.")
    else:
        print("  Reference links CSV not found under setup/ttc_route_29_data/,")
        print("  so the placeholder cannot be reconstructed on this checkout.")
        print(f"  For reference, the OLD formula was: tt_mean * {PLACEHOLDER_SPEED_KMH:.0f} km/h.")

    # --- persist outputs ---------------------------------------------------- #
    C.banner("SAVING OUTPUTS")
    out_cols = ["stop_seq", "StopID", "ONSTREET", "ATSTREET", "Lat", "Lon",
                "cum_dist_m", "spacing_real_m", "placeholder_m"]
    C.save_csv(stops[out_cols], f"day2_geometry_{direction.lower()}.csv")

    fig = _plot_route_map(geo["stops"], direction)
    C.savefig(fig, f"day2_route_map_{direction.lower()}.png")
    plt.close(fig)

    # STUDENT TODO (branches): C.stop_sequence collapses Route 29's branches
    # (DLWI, 29Dcon, DLPRcon) onto one median order. Branch-only stops therefore
    # get slotted into a single linear chain, which can inflate/deflate a link if
    # a branch skips a stop. If you care about branch fidelity, derive geometry
    # per Branch (filter apc_clean/apc_raw on the 'Branch' column) and reconcile
    # the common trunk. See docs/04_week_plan.md (Day 2).
    #
    # STUDENT TODO (GTFS cross-check): GTFS shapes.txt + stop_times.txt carry
    # shape_dist_traveled, an independent along-route distance. You can validate
    # this haversine spacing against it — BUT the GTFS feed is a 2020 snapshot
    # while the APC is Nov-2023, and only ~72% of 2023 stop_ids appear in the
    # 2020 stops.txt (GAP 4). Use GTFS as a sanity cross-check / source of the
    # smooth shape polyline, NOT as ground truth for the 2023 stop set. Load it
    # via C.load_gtfs('shapes'), C.load_gtfs('stop_times'), C.load_gtfs('trips').


def main() -> None:
    # Optional CLI arg restricts to one direction; default processes both.
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    directions = [args[0]] if args else list(C.DIRECTIONS)
    for d in directions:
        _run_one(d)


if __name__ == "__main__":
    main()
