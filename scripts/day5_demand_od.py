"""
day5_demand_od.py — per-stop demand (lambda) and an origin-destination (OD) matrix
for TTC Route 29, derived from the real November-2023 APC boarding/alighting counts.

WHAT THIS SCRIPT PRODUCES
-------------------------
1. Per-stop arrival rate  lambda_i  (passengers / second) at every stop.
2. A full origin->destination RATE matrix (passengers / second), estimated by
   Iterative Proportional Fitting (IPF, a.k.a. "biproportional" or "Furness"
   balancing) from the marginal boarding and alighting totals.

WHY THIS MATTERS (read docs/03_data_to_env_mapping.md → GAP 5)
--------------------------------------------------------------
The reference file setup/ttc_route_29_data/stops_north.csv computed
    lambda_i = avg_boarding_per_trip_i / 300
where 300 s is the HARD-CODED placeholder dispatching headway (GAP 1). The real
measured headway (Day 4) is ~536 s northbound — roughly *double*. Because
lambda is boardings-per-trip divided by the headway, fixing the headway rescales
EVERY lambda. That coupling is the whole point of GAP 5, and it is made explicit
below: `derive_demand_od` takes `headway_sec` and, if you do not pass one, it
imports Day 4's measured headway rather than silently reusing 300.

HOW THE OD MATRIX IS BUILT
--------------------------
We only observe *marginals*: how many people board at each stop (O_i) and how
many alight at each stop (D_j). We do NOT observe who-went-where. IPF is the
standard, defensible way to reconstruct a full OD table from those marginals
plus a feasibility structure:
    * Feasibility: on a one-way line a passenger can only travel DOWNSTREAM, so
      the OD matrix must be strictly upper-triangular in canonical stop order.
    * IPF alternately rescales rows to hit the boarding totals O_i and columns
      to hit the alighting totals D_j, converging to the unique matrix that
      matches both margins while respecting the zeros of the feasibility mask.

Run it standalone (from the repo root /home/jiahao/Documents/busoperation):
    python student_project/scripts/day5_demand_od.py

Importable, side-effect-free entry point (used by build_env_data.py and tests):
    derive_demand_od(direction, headway_sec=None) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import common` and `import day4_headway` work no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set BEFORE importing pyplot
import matplotlib.pyplot as plt

import common as C

# The old placeholder headway that the reference stops_north.csv divided by.
# Kept only so we can print OLD-vs-NEW and show the student the improvement.
PLACEHOLDER_HEADWAY_SEC = 300.0


# --------------------------------------------------------------------------- #
# Step 1 — marginal demand: boardings & alightings per stop, in canonical order
# --------------------------------------------------------------------------- #
def _load_stop_stats(direction: str) -> pd.DataFrame:
    """Return per-stop demand marginals in canonical (terminal..terminal) order.

    Columns: StopID(str), stop_seq, boardings, alightings, n_trips.
    `boardings`/`alightings` are November-2023 totals; `n_trips` is the number
    of distinct trips that served the stop (so boardings/n_trips = per-trip mean).
    """
    apc = C.load_apc_clean()

    # Canonical stop order (median StopSeq, robust to branches) comes from common.
    seq = C.stop_sequence(apc, direction)[["StopID", "stop_seq"]].copy()

    # Sum the raw counts per stop for this direction.
    d = C.route_dir(apc, direction).copy()
    d["StopID"] = d["StopID"].astype(str)
    grp = (
        d.groupby("StopID")
        .agg(
            boardings=("Boarding", "sum"),
            alightings=("Alighting", "sum"),
            n_trips=("TripID", "nunique"),
        )
        .reset_index()
    )

    # Merge onto the canonical order so row order == node order everywhere.
    stats = seq.merge(grp, on="StopID", how="left")
    for col in ("boardings", "alightings", "n_trips"):
        stats[col] = stats[col].fillna(0.0)
    stats = stats.sort_values("stop_seq").reset_index(drop=True)
    return stats


# --------------------------------------------------------------------------- #
# Step 2 — the IPF engine
# --------------------------------------------------------------------------- #
def ipf(seed: np.ndarray,
        row_margins: np.ndarray,
        col_margins: np.ndarray,
        iters: int = 50,
        tol: float = 1e-9) -> np.ndarray:
    """Iterative Proportional Fitting (Furness balancing).

    Find a matrix M with the same zero-pattern as `seed` whose row sums match
    `row_margins` and whose column sums match `col_margins`. Each iteration:
        1. scale every row so its sum equals the target row margin;
        2. scale every column so its sum equals the target column margin.
    Rows/columns whose current sum is 0 (fully masked out by the seed) are left
    at 0 — their target margin is simply infeasible and cannot be allocated.

    Parameters
    ----------
    seed : (N, N) array — non-negative; zeros are structural (stay zero forever).
    row_margins, col_margins : length-N target sums. For a consistent solution
        their totals should be equal; we assume the caller has rescaled them.
    iters : max sweeps. col : max sweeps.
    tol : stop early once the largest margin error falls below this.
    """
    M = np.array(seed, dtype=float)
    row_margins = np.asarray(row_margins, dtype=float)
    col_margins = np.asarray(col_margins, dtype=float)

    for _ in range(iters):
        # --- row scaling: make each row sum to its boarding total ---
        rs = M.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rf = np.where(rs > 0, row_margins / rs, 0.0)
        M *= rf[:, None]

        # --- column scaling: make each column sum to its alighting total ---
        cs = M.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cf = np.where(cs > 0, col_margins / cs, 0.0)
        M *= cf[None, :]

        # --- convergence: largest deviation from either target margin ---
        row_err = np.abs(M.sum(axis=1) - row_margins)[row_margins > 0]
        col_err = np.abs(M.sum(axis=0) - col_margins)[col_margins > 0]
        max_err = max(
            float(row_err.max()) if row_err.size else 0.0,
            float(col_err.max()) if col_err.size else 0.0,
        )
        if max_err < tol:
            break

    return M


# --------------------------------------------------------------------------- #
# Step 3 — the public entry point
# --------------------------------------------------------------------------- #
def derive_demand_od(direction: str, headway_sec: float | None = None) -> dict:
    """Derive per-stop lambda and a full OD RATE matrix for one direction.

    Parameters
    ----------
    direction : "NORTH" or "SOUTH" (case-insensitive).
    headway_sec : dispatching headway in seconds used to convert per-trip
        boardings into a per-second arrival rate. If None, Day 4's *measured*
        overall headway is imported and used (NOT the 300 s placeholder).

    Returns
    -------
    dict with keys:
        "lambda"     : {stop_id(str): pax/sec}  (== per-stop arrival rate)
        "od_rate"    : DataFrame[index=origin, cols=dest] pax/sec, strictly
                       upper-triangular, each origin row summing to that lambda.
        "boardings"  : {stop_id: total boardings (Nov 2023)}
        "alightings" : {stop_id: total alightings (Nov 2023)}
        "headway_sec": the headway actually used (float) — handy for provenance.
    """
    direction = direction.upper()

    # ---- GAP 5, made explicit -------------------------------------------- #
    # lambda_i = (mean boardings per trip at i) / headway_sec.
    # The reference data divided by the 300 s PLACEHOLDER; here we divide by the
    # measured headway from Day 4, so changing the headway rescales every lambda.
    if headway_sec is None:
        import day4_headway  # lazy import: only needed when caller omits headway
        headway_sec = float(day4_headway.derive_headway(direction)["overall"][0])
    headway_sec = float(headway_sec)
    if headway_sec <= 0:
        raise ValueError(f"headway_sec must be positive, got {headway_sec}")

    stats = _load_stop_stats(direction)
    order = stats["StopID"].tolist()             # canonical terminal..terminal order
    N = len(order)

    boardings = stats["boardings"].to_numpy(dtype=float)
    alightings = stats["alightings"].to_numpy(dtype=float)
    n_trips = stats["n_trips"].to_numpy(dtype=float)

    # Per-trip mean boardings, then per-second rate. Guard n_trips == 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        board_per_trip = np.where(n_trips > 0, boardings / n_trips, 0.0)
    lam = board_per_trip / headway_sec           # pax / sec at each stop

    # ---- OD via IPF ------------------------------------------------------ #
    # Row margins  O_i = total boardings at i.
    # Col margins  D_j = total alightings at j, rescaled so sum(D) == sum(O)
    #              (IPF needs consistent totals to have an exact solution).
    O = boardings.copy()
    D = alightings.copy()
    if D.sum() > 0:
        D = D * (O.sum() / D.sum())

    # Feasibility mask: passengers only travel DOWNSTREAM, so entry (i, j) is
    # allowed iff origin i is strictly upstream of dest j. In canonical order
    # "upstream" means a smaller index -> strictly upper-triangular.
    mask = np.triu(np.ones((N, N), dtype=float), k=1)

    counts = ipf(mask, O, D, iters=50)           # OD COUNT matrix (people/month)

    # ---- Convert counts -> RATE, consistent with lambda ------------------ #
    # Distribute each origin's lambda_i across destinations using the IPF row
    # proportions. This is algebraically the same as "divide the count row by
    # n_trips_i and by headway_sec" whenever the IPF row sum equals O_i, but it
    # is numerically exact: every feasible origin row then sums to lambda_i.
    rate = np.zeros((N, N), dtype=float)
    row_sums = counts.sum(axis=1)
    for i in range(N):
        if lam[i] <= 0:
            continue                             # nobody boards here -> zero row
        if row_sums[i] > 0:
            rate[i, :] = lam[i] * counts[i, :] / row_sums[i]
        else:
            # Origin has feasible destinations but IPF gave it zero mass (e.g.
            # all downstream alighting margins were zero). Fall back to spreading
            # lambda_i uniformly over the feasible (downstream) destinations so
            # the row still sums to lambda_i. Origins with NO downstream stop
            # (the final terminal) legitimately stay all-zero.
            feasible = mask[i, :] > 0
            k = int(feasible.sum())
            if k > 0:
                rate[i, feasible] = lam[i] / k

    od_rate = pd.DataFrame(rate, index=order, columns=order)

    return {
        "lambda": {sid: float(lam[i]) for i, sid in enumerate(order)},
        "od_rate": od_rate,
        "boardings": {sid: float(boardings[i]) for i, sid in enumerate(order)},
        "alightings": {sid: float(alightings[i]) for i, sid in enumerate(order)},
        "headway_sec": headway_sec,
    }


# --------------------------------------------------------------------------- #
# Reporting / standalone run
# --------------------------------------------------------------------------- #
def _plot_od_heatmap(od_rate: pd.DataFrame, direction: str) -> None:
    """Save a heatmap of the OD rate matrix (log color so small flows show)."""
    order = list(od_rate.index)
    M = od_rate.to_numpy()
    shown = np.where(M > 0, M, np.nan)           # hide structural zeros

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(np.log10(shown), aspect="auto", cmap="viridis",
                   origin="upper", interpolation="nearest")
    ax.set_title(f"Route 29 {direction} — OD rate matrix "
                 f"(log10 pax/sec, {len(order)} stops)")
    ax.set_xlabel("destination (stop order →)")
    ax.set_ylabel("origin (stop order →)")

    # Sparse ticks so the labels stay readable.
    step = max(1, len(order) // 12)
    ticks = list(range(0, len(order), step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([order[t] for t in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([order[t] for t in ticks], fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("log10(pax / sec)")
    C.savefig(fig, f"day5_od_heatmap_{direction.lower()}.png")
    plt.close(fig)


def _report(direction: str) -> None:
    """Full Day-5 report for one direction: derive, print, verify, save, plot."""
    direction = direction.upper()
    C.banner(f"DAY 5 — DEMAND & OD  |  Route 29 {direction}")

    res = derive_demand_od(direction)            # headway_sec None -> Day 4 value
    headway_sec = res["headway_sec"]
    lam = res["lambda"]
    boardings = res["boardings"]
    alightings = res["alightings"]
    od = res["od_rate"]
    order = list(od.index)

    # ---- GAP 5 headline: OLD placeholder vs NEW measured headway --------- #
    print(f"\nHeadway used for lambda: {headway_sec:.1f} s "
          f"(measured, Day 4)   vs   OLD placeholder {PLACEHOLDER_HEADWAY_SEC:.0f} s")
    print(f"  => every lambda is scaled by {PLACEHOLDER_HEADWAY_SEC/headway_sec:.3f} "
          f"relative to the reference stops_{direction.lower()}.csv values.")

    # ---- per-stop lambda head ------------------------------------------- #
    print("\nPer-stop arrival rate (first 8 stops):")
    print(f"  {'seq':>3}  {'stop_id':>8}  {'board/trip':>10}  "
          f"{'lam_new(/s)':>12}  {'lam_old(/s)':>12}")
    for i, sid in enumerate(order[:8]):
        bpt = boardings[sid] / max(1.0, res_ntrips(direction).get(sid, 1.0))
        lam_old = bpt / PLACEHOLDER_HEADWAY_SEC
        print(f"  {i+1:>3}  {sid:>8}  {bpt:>10.4f}  "
              f"{lam[sid]:>12.6f}  {lam_old:>12.6f}")

    # ---- system-level totals -------------------------------------------- #
    total_rate = sum(lam.values())               # pax / sec boarding the route
    print(f"\nTotal system boarding rate: {total_rate:.5f} pax/sec "
          f"= {total_rate*3600:.1f} pax/hour")
    print(f"Total Nov-2023 boardings : {sum(boardings.values()):,.0f}  |  "
          f"alightings: {sum(alightings.values()):,.0f}")

    # ---- verifications --------------------------------------------------- #
    M = od.to_numpy()
    non_negative = bool((M >= 0).all())
    lower_tri_zero = bool(np.allclose(np.tril(M), 0.0))   # zero on/below diagonal
    print("\nOD matrix checks:")
    print(f"  non-negative                : {non_negative}")
    print(f"  strictly upper-triangular   : {lower_tri_zero}")

    # Each origin row-sum should equal that origin's lambda (feasible origins).
    row_sums = M.sum(axis=1)
    lam_vec = np.array([lam[s] for s in order])
    has_downstream = np.arange(len(order)) < (len(order) - 1)  # all but last stop
    diffs = np.abs(row_sums - lam_vec)[has_downstream]
    max_diff = float(diffs.max()) if diffs.size else 0.0
    print(f"  max |row_sum - lambda|      : {max_diff:.3e}  "
          f"(over origins with a downstream stop)")
    last = order[-1]
    print(f"  note: final terminal {last} has no downstream stop, so its OD row "
          f"is all zeros while lambda={lam[last]:.2e} (boardings there are "
          f"unallocatable).")

    # ---- hard invariants: fail loudly if the OD matrix is malformed ------- #
    # These asserts make the guarantees in docs/04_week_plan.md real: if any
    # ever trips, derive_demand_od produced something the simulator must not be
    # fed. (The test suite re-checks the same properties from the outside.)
    assert non_negative, "OD rate matrix has negative entries"
    assert lower_tri_zero, \
        "OD rate matrix is not strictly upper-triangular (passengers must go downstream)"
    assert max_diff < 1e-6, \
        f"OD origin row-sums do not match per-stop lambda (max |diff| = {max_diff:.3e})"

    # ---- save artifacts -------------------------------------------------- #
    nt = res_ntrips(direction)
    lam_df = pd.DataFrame({
        "stop_id": order,
        "stop_seq": range(1, len(order) + 1),
        "boardings": [boardings[s] for s in order],
        "alightings": [alightings[s] for s in order],
        "n_trips": [nt.get(s, 0.0) for s in order],
        "avg_boarding_per_trip": [boardings[s] / max(1.0, nt.get(s, 1.0)) for s in order],
        "headway_sec": headway_sec,
        "lambda_pax_sec": [lam[s] for s in order],
        "lambda_pax_min": [lam[s] * 60.0 for s in order],
    })
    C.save_csv(lam_df, f"day5_lambda_{direction.lower()}.csv")
    C.save_csv(od, f"day5_od_rate_matrix_{direction.lower()}.csv", index=True)
    _plot_od_heatmap(od, direction)


# Small cache so the report does not re-read the APC CSV just for trip counts.
_NTRIPS_CACHE: dict[str, dict[str, float]] = {}


def res_ntrips(direction: str) -> dict[str, float]:
    """{stop_id: n_trips} for a direction (cached; used only for pretty-printing)."""
    direction = direction.upper()
    if direction not in _NTRIPS_CACHE:
        stats = _load_stop_stats(direction)
        _NTRIPS_CACHE[direction] = dict(
            zip(stats["StopID"].astype(str), stats["n_trips"].astype(float))
        )
    return _NTRIPS_CACHE[direction]


def main() -> None:
    # STUDENT TODO: alternative OD estimators.
    #   IPF assumes the OD split at each origin is proportional to the downstream
    #   alighting margins. Other defensible choices you could implement and
    #   compare: (a) a distance-decay gravity model rate_ij ∝ O_i*D_j*f(dist_ij);
    #   (b) a "closest reasonable alighting" heuristic; (c) fitting a small
    #   trip-length distribution. Compare the resulting mean trip length against
    #   the AVL evidence before trusting any of them.
    #
    # STUDENT TODO: time-of-day OD.
    #   Everything here pools all of November. PeriodID (AM peak / Midday / PM
    #   peak / ...) lets you build one OD matrix per period, each with its own
    #   period headway from Day 4. Loop over PeriodID, filter the APC rows, and
    #   emit od_rate_<dir>_<period>.csv. The simulator can then swing demand by
    #   time of day instead of using a single flat matrix.
    for direction in C.DIRECTIONS:
        _report(direction)

    C.banner("DAY 5 DONE")
    print("Next: build_env_data.py stitches Day 2 (geometry), Day 3 (travel time),")
    print("Day 4 (headway) and Day 5 (demand/OD) into a DataLoader-compatible bundle.")


if __name__ == "__main__":
    main()
