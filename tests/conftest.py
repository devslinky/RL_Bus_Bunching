"""
conftest.py — shared pytest configuration for the TTC Route 29 test suite.

This file does two jobs, and pytest imports it automatically before it collects
any test in this directory:

1. It fixes ``sys.path`` so that every test can simply write::

       import common as C
       import day2_geometry
       import build_env_data
       from setup.blueprint import Blueprint

   without any per-file path gymnastics. We add the *scripts* directory (so the
   ``common`` / ``dayN_*`` / ``build_env_data`` modules resolve as top-level
   imports, exactly the way the scripts import each other) and the *repo root*
   (so ``setup.blueprint`` and the rest of the simulator package resolve).

2. It provides a session-scoped ``derived_north`` fixture that runs the full
   NORTH derivation pipeline **once** and hands the resulting dict to every test
   that asks for it. Deriving the environment reads a few hundred-thousand rows
   of APC data, so recomputing it per-test would make the suite crawl.

Student notes
-------------
* Heavy imports (pandas, the derivation scripts) happen *inside* the fixture,
  not at module import time, so ``pytest --collect-only`` stays fast.
* All paths are computed from ``__file__`` so the suite works no matter what
  directory you launch ``pytest`` from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Path wiring
# --------------------------------------------------------------------------- #
# conftest.py lives at:  .../student_project/tests/conftest.py
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent                    # .../student_project
SCRIPTS_DIR = PROJECT_DIR / "scripts"             # common.py, dayN_*.py, build_env_data.py

# CODE_ROOT holds raw_data/ + setup/ + simulator/ + agent/. student_project bundles its own
# copies (self-contained); fall back to an enclosing repo if they are absent. This mirrors
# the detection in scripts/common.py.
if (PROJECT_DIR / "raw_data").is_dir() and (PROJECT_DIR / "setup").is_dir():
    CODE_ROOT = PROJECT_DIR
else:
    CODE_ROOT = PROJECT_DIR.parent
REPO_ROOT = CODE_ROOT                             # alias kept for older references


def _prepend(path: Path) -> None:
    """Put ``path`` at the front of ``sys.path`` exactly once (idempotent)."""
    s = str(path)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


# CODE_ROOT first so `import setup.* / simulator.* / agent.*` resolve to the bundled copies;
# scripts dir on top so `import common` / `import day2_geometry` resolve as top-level modules.
_prepend(CODE_ROOT)
_prepend(SCRIPTS_DIR)

# Some env code (e.g. setup/chengdu_route_3_data/dataloader.py) opens data by a cwd-relative
# path, so `import setup.blueprint` only works when cwd holds setup/. Anchor cwd at CODE_ROOT
# so the suite runs from anywhere.
os.chdir(CODE_ROOT)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def derived_north():
    """Run ``build_env_data.derive_all("NORTH")`` once and share the result.

    Returns the DataLoader-compatible dict (node_ids, spacing, link_time_info,
    od_rate_table, dispatching_headway, stop_pax_arrival_rate, ...). Session
    scope means the expensive derivation runs a single time for the whole test
    run; every test that lists ``derived_north`` as an argument reuses the same
    object.

    The import is done lazily so that merely *collecting* tests (which does not
    touch this fixture) never pays the cost of importing pandas or reading the
    raw CSVs.
    """
    import build_env_data  # noqa: WPS433 (intentional local import — keeps collection fast)

    return build_env_data.derive_all("NORTH")
