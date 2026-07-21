"""
common.py — shared utilities for the TTC Route 29 data-to-environment project.

Every `dayN_*.py` script and `build_env_data.py` imports this module so that
paths, loaders, and helpers are defined in exactly one place.

Usage inside a sibling script:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C

    apc = C.load_apc_clean()

Design notes for the student
----------------------------
* Nothing here writes into `setup/`. Every output goes to `student_project/outputs/`
  so you can never damage the working simulator while experimenting.
* The loaders are thin wrappers around `pandas.read_csv` — read the source CSVs
  directly whenever you want to understand a column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# common.py lives at:  <root>/student_project/scripts/common.py
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent                 # .../student_project
_ENCLOSING = PROJECT_DIR.parent                  # the repo it may be embedded in

# student_project is designed to be SELF-CONTAINED: it bundles its own copies of
# raw_data/, setup/, simulator/, agent/, config.py, quick_run.py. Prefer those (so the
# folder works when handed over on its own); fall back to an enclosing repo if they are
# absent. CODE_ROOT is the directory that holds raw_data/ + setup/ + simulator/ + agent/.
if (PROJECT_DIR / "raw_data").is_dir() and (PROJECT_DIR / "setup").is_dir():
    CODE_ROOT = PROJECT_DIR                       # self-contained (recommended)
else:
    CODE_ROOT = _ENCLOSING                        # embedded in a larger repo
REPO_ROOT = CODE_ROOT                             # alias kept for older references

RAW = CODE_ROOT / "raw_data"                      # source AVL / APC / GTFS data
SETUP_DATA = CODE_ROOT / "setup" / "ttc_route_29_data"   # the *reference* derived data
OUT = PROJECT_DIR / "outputs"                     # everything you generate lands here
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Constants (grounded in the real data — see docs/02_data_dictionary.md)
# --------------------------------------------------------------------------- #
ROUTE = 29
DIRECTIONS = ("NORTH", "SOUTH")

# Raw file names, kept in one place so a rename only happens here.
F_APC_CLEAN = "ttc_apc_clean_data.csv"   # per stop / per trip boarding, alighting, load, dwell
F_APC_RAW = "ttc_apc_data.csv"           # same idea + per-stop Lat/Lon (geometry source!)
F_AVL_SEG = "ttc_avl_seg_clean_data.csv" # cleaned GPS pings with TripID, KPH, RouteDirection
F_AVL_RAW = "ttc_avl_data.csv"           # raw GPS pings (large, ~6.2M rows)
GTFS_FILES = ("stops.txt", "trips.txt", "stop_times.txt", "shapes.txt",
              "routes.txt", "calendar.txt")

# The GTFS route_id that corresponds to surface route "29 DUFFERIN".
GTFS_ROUTE_29_ID = 61327


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_apc_clean() -> pd.DataFrame:
    """Cleaned APC table: one row per (trip, stop) with Boarding/Alighting/loads."""
    return pd.read_csv(RAW / F_APC_CLEAN, low_memory=False)


def load_apc_raw() -> pd.DataFrame:
    """Raw APC table. Unlike the clean one, this carries per-stop Lat/Lon."""
    return pd.read_csv(RAW / F_APC_RAW, low_memory=False)


def load_avl_seg() -> pd.DataFrame:
    """Cleaned, segmented AVL/GPS pings (TripID, Distance, KPH, RouteDirection)."""
    return pd.read_csv(RAW / F_AVL_SEG, low_memory=False)


def load_gtfs(name: str) -> pd.DataFrame:
    """Load a GTFS text table by file name, e.g. load_gtfs('stops.txt')."""
    if not name.endswith(".txt"):
        name = name + ".txt"
    return pd.read_csv(RAW / name, low_memory=False)


# --------------------------------------------------------------------------- #
# Filtering / sequencing helpers
# --------------------------------------------------------------------------- #
def route_dir(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Return the Route-29 rows for a single direction ('NORTH' or 'SOUTH')."""
    direction = direction.upper()
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    out = df.copy()
    if "Route" in out.columns:
        out = out[out["Route"] == ROUTE]
    return out[out["RouteDirection"].str.upper() == direction]


def stop_sequence(apc_clean: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Canonical ordered stop list for one direction.

    Each physical stop appears once, ordered by its *median* StopSeq across all
    trips (robust to occasional mis-sequenced trips and to branch differences).

    Returns a DataFrame with columns:
        StopID, stop_seq, ONSTREET, ATSTREET, n_trips
    ordered from terminal (seq 1) to terminal (seq N).

    NOTE FOR THE STUDENT: Route 29 has several branches (DLWI, 29Dcon, DLPRcon)
    that do not all serve the same stops. Using the median StopSeq is a simple
    canonicalization; deciding how to treat branch-specific stops is a real
    modelling choice — see docs/04_week_plan.md (Day 2).
    """
    d = route_dir(apc_clean, direction)
    grp = d.groupby("StopID")
    seq = (
        grp.agg(
            stop_seq=("StopSeq", "median"),
            ONSTREET=("ONSTREET", "first"),
            ATSTREET=("ATSTREET", "first"),
            n_trips=("TripID", "nunique"),
        )
        .reset_index()
        .sort_values("stop_seq")
        .reset_index(drop=True)
    )
    seq["StopID"] = seq["StopID"].astype(str)
    return seq


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in METERS."""
    R = 6_371_000.0  # Earth radius, meters
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def stop_latlon(apc_raw: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Median Lat/Lon per stop for one direction (from the raw APC table).

    Returns columns: StopID, Lat, Lon.  Coverage on Route 29 is ~100%.
    """
    d = route_dir(apc_raw, direction).dropna(subset=["Lat", "Lon"])
    ll = (
        d.groupby("StopID")
        .agg(Lat=("Lat", "median"), Lon=("Lon", "median"))
        .reset_index()
    )
    ll["StopID"] = ll["StopID"].astype(str)
    return ll


# --------------------------------------------------------------------------- #
# Output helpers  (all writes go to student_project/outputs/)
# --------------------------------------------------------------------------- #
def out_path(name: str) -> Path:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    p = out_path(name)
    df.to_csv(p, index=index)
    print(f"  [saved] {p.relative_to(REPO_ROOT)}  ({len(df)} rows)")
    return p


def save_json(obj: dict, name: str) -> Path:
    p = out_path(name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  [saved] {p.relative_to(REPO_ROOT)}")
    return p


def savefig(fig, name: str) -> Path:
    p = out_path(name)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    print(f"  [saved] {p.relative_to(REPO_ROOT)}")
    return p


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


if __name__ == "__main__":
    # Smoke check: print where everything is and confirm the raw files exist.
    banner("common.py path check")
    mode = "self-contained" if CODE_ROOT == PROJECT_DIR else "embedded in enclosing repo"
    print(f"mode        = {mode}")
    print(f"CODE_ROOT   = {CODE_ROOT}  (holds raw_data/ setup/ simulator/ agent/)")
    print(f"RAW         = {RAW}  (exists={RAW.exists()})")
    print(f"SETUP_DATA  = {SETUP_DATA}  (exists={SETUP_DATA.exists()})")
    print(f"OUT         = {OUT}")
    print("\nRaw files present:")
    for fn in (F_APC_CLEAN, F_APC_RAW, F_AVL_SEG, *GTFS_FILES):
        print(f"  [{'ok' if (RAW / fn).exists() else 'MISSING'}] {fn}")
