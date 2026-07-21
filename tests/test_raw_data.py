"""
test_raw_data.py — sanity checks on the RAW inputs.

Run this FIRST, before you write or trust any derivation script. If these
tests are red, nothing downstream can be right: the ETL you build in
``scripts/dayN_*.py`` and ``scripts/build_env_data.py`` reads exactly these
files. These tests answer one question: "are the raw TTC Route 29 inputs
present and well-formed?"

Run it with::

    cd /home/jiahao/Documents/busoperation
    pytest student_project/tests/test_raw_data.py -v

What "done" looks like: every test passes. If a file is MISSING, the test
tells you which one (see docs/02_data_dictionary.md for what each file is and
where it comes from). We do NOT check the giant raw AVL file
(``ttc_avl_data.csv``) or the 2020 GTFS ``calendar.txt`` here because the
pipeline does not depend on them (see GAP 4 in docs/03_data_to_env_mapping.md).

Speed notes:
* Column-existence checks read only a few thousand rows (``nrows=``) — fast.
* The "Route 29 / NORTH+SOUTH present" check reads the WHOLE cleaned APC file,
  but only two columns (``usecols=``), so it is still quick.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the shared helper module importable no matter where pytest is invoked
# from. common.py lives in ../scripts relative to this test file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import common as C  # noqa: E402


# --------------------------------------------------------------------------- #
# What we expect to find on disk
# --------------------------------------------------------------------------- #
# Every raw file the derivation pipeline actually reads. Keyed by the constant
# on common.py where one exists so a rename only ever happens in common.py.
REQUIRED_RAW_FILES = [
    C.F_APC_CLEAN,           # per (trip, stop): boarding/alighting/load/dwell + times
    C.F_APC_RAW,             # same idea, PLUS per-stop Lat/Lon (geometry source)
    C.F_AVL_SEG,             # cleaned segmented GPS pings (KPH cross-check)
    "stops.txt",             # GTFS: stop_id -> lat/lon/name (cross-check only)
    "trips.txt",             # GTFS: trip -> route/direction/shape
    "stop_times.txt",        # GTFS: per-trip stop sequence + shape_dist_traveled
    "shapes.txt",            # GTFS: route polylines
    "routes.txt",            # GTFS: route_id 61327 == "29 DUFFERIN"
]

# The columns each dayN script relies on. Subset check (the real files carry
# more columns); we only assert the ones the pipeline consumes.
APC_CLEAN_REQUIRED_COLS = [
    "DAYTYPE", "Route", "Branch", "RouteDirection", "StopID", "StopSeq",
    "StopArrivalTime", "StopDepartureTime", "Boarding", "Alighting",
    "Arrload", "Depload", "PeriodID", "TripID", "AvgDwell",
]
APC_RAW_REQUIRED_COLS = ["Lat", "Lon"]          # the whole point of the raw APC file
AVL_SEG_REQUIRED_COLS = ["TripID", "KPH", "RouteDirection"]


def _read_header(path: Path, nrows: int = 2000) -> pd.DataFrame:
    """Read just the first few rows of a CSV so we can inspect its columns fast."""
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _assert_has_columns(df: pd.DataFrame, required, where: str) -> None:
    """Fail with a helpful message listing exactly which columns are missing."""
    missing = [c for c in required if c not in df.columns]
    assert not missing, (
        f"{where} is missing required column(s): {missing}. "
        f"Columns present: {list(df.columns)}"
    )


# --------------------------------------------------------------------------- #
# Existence
# --------------------------------------------------------------------------- #
def test_raw_dir_exists():
    """The raw_data/ directory must exist (see common.C.RAW)."""
    assert C.RAW.exists() and C.RAW.is_dir(), (
        f"raw data directory not found: {C.RAW}. "
        "Check that raw_data/ sits next to student_project/ in the repo root."
    )


@pytest.mark.parametrize("filename", REQUIRED_RAW_FILES)
def test_required_raw_file_exists(filename):
    """Each raw file the pipeline reads must be present and non-empty."""
    path = C.RAW / filename
    assert path.exists(), (
        f"missing raw file: {path}. "
        "See docs/02_data_dictionary.md for what this file is and where to get it."
    )
    assert path.stat().st_size > 0, f"raw file is empty (0 bytes): {path}"


# --------------------------------------------------------------------------- #
# Column schemas
# --------------------------------------------------------------------------- #
def test_apc_clean_has_required_columns():
    """Cleaned APC file must carry every column the demand/headway/tt code uses."""
    df = _read_header(C.RAW / C.F_APC_CLEAN)
    _assert_has_columns(df, APC_CLEAN_REQUIRED_COLS, C.F_APC_CLEAN)


def test_apc_raw_has_latlon():
    """Raw APC file must carry Lat/Lon — it is our geometry source (GAP 2)."""
    df = _read_header(C.RAW / C.F_APC_RAW)
    _assert_has_columns(df, APC_RAW_REQUIRED_COLS, C.F_APC_RAW)


def test_avl_seg_has_required_columns():
    """Segmented AVL file must carry TripID, KPH, RouteDirection for speed checks."""
    df = _read_header(C.RAW / C.F_AVL_SEG)
    _assert_has_columns(df, AVL_SEG_REQUIRED_COLS, C.F_AVL_SEG)


# --------------------------------------------------------------------------- #
# Content: Route 29 and both directions are actually present
# --------------------------------------------------------------------------- #
def test_route_29_and_both_directions_present():
    """The cleaned APC file must contain Route 29 with both NORTH and SOUTH.

    We read the whole file, but only two columns, so this stays fast.
    """
    df = pd.read_csv(
        C.RAW / C.F_APC_CLEAN,
        usecols=["Route", "RouteDirection"],
        low_memory=False,
    )

    routes = set(df["Route"].dropna().unique())
    assert C.ROUTE in routes, (
        f"Route {C.ROUTE} not found in {C.F_APC_CLEAN}; routes present: {routes}"
    )

    directions = {str(d).upper() for d in df["RouteDirection"].dropna().unique()}
    for want in C.DIRECTIONS:  # ("NORTH", "SOUTH")
        assert want in directions, (
            f"RouteDirection {want!r} not found in {C.F_APC_CLEAN}; "
            f"directions present: {directions}"
        )


def test_route_dir_helper_returns_rows_for_each_direction():
    """C.route_dir() must actually return Route-29 rows for NORTH and SOUTH.

    This exercises the helper the whole pipeline leans on, using only the two
    columns needed so the read stays cheap.
    """
    df = pd.read_csv(
        C.RAW / C.F_APC_CLEAN,
        usecols=["Route", "RouteDirection"],
        low_memory=False,
    )
    for direction in C.DIRECTIONS:
        sub = C.route_dir(df, direction)
        assert len(sub) > 0, (
            f"C.route_dir(..., {direction!r}) returned no rows; "
            "expected Route-29 trips in that direction."
        )
        # Every returned row is Route 29 in the requested direction.
        assert (sub["Route"] == C.ROUTE).all()
        assert (sub["RouteDirection"].str.upper() == direction).all()


if __name__ == "__main__":
    # Allow running the checks directly:  python student_project/tests/test_raw_data.py
    sys.exit(pytest.main([__file__, "-v"]))
