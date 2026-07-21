"""
day6_integrate.py — CAPSTONE companion tool (it does NOT do the integration for you).

Days 1-5 produced a DataLoader-compatible bundle in
`student_project/outputs/derived/{north,south}/`. Day 6 is where YOU make the real
`simulator/` run on it. This is deliberately YOUR design work — see
docs/06_integrate_into_simulator.md for the contract, the two placeholders to remove, and
the design options. This script only *helps*; it never writes a solution into `setup/`.

What it gives you (no spoilers, no edits to setup/):
  --preview  (default) : print OLD (shipped placeholder) vs NEW (your derived data) for
                         headway and spacing, and a suggested run config. This shows you the
                         TARGET numbers your integration should make the env report.
  --check              : after YOU have built your real env and registered it in
                         setup/blueprint.py, validate that Blueprint(<your env>) actually
                         uses the real headway/spacing. Pass/fail on YOUR work.
  --run-sim            : run a short DoNothing episode on your registered env (needs the
                         torch/agent stack, same as quick_run.py).

Env-name convention: register your env as `ttc_route_29_<direction>_real` so --check and
--run-sim can find it (or pass --env-name).

Run from the project root (the folder that holds scripts/, setup/, raw_data/):
    python scripts/day6_integrate.py                 # if student_project is your root, or
    python student_project/scripts/day6_integrate.py # if it is embedded in the repo
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# So `import setup.*` / `simulator.*` / `agent.*` resolve to the bundled copies.
sys.path.insert(0, str(C.CODE_ROOT))

DERIVED = C.OUT / "derived"                       # written by build_env_data.py
PLACEHOLDER_HEADWAY = (300.0, 60.0)               # the value in the shipped dataloader


# --------------------------------------------------------------------------- #
# Helpers (read the derived bundle + the shipped placeholder loader; no env wiring)
# --------------------------------------------------------------------------- #
def _load_bundle(direction: str) -> dict:
    p = DERIVED / direction.lower() / f"data_{direction.lower()}.pickle"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run build_env_data.py first:\n"
            f"    python {C.PROJECT_DIR.name}/scripts/build_env_data.py"
        )
    with open(p, "rb") as f:
        return pickle.load(f)


def suggest_fleet_size(direction: str) -> dict:
    """Estimate fleet_size ~= round-trip time / headway, from YOUR derived numbers.

    round_trip ~= 2 * (sum of link travel times + ~15 s dwell per stop). Purely a config
    suggestion — it does not integrate anything.
    """
    b = _load_bundle(direction)
    link_tt = sum(v["loc"] for v in b["link_time_info"].values())   # one-way run (s)
    n_stops = len(b["node_ids"])
    one_way = link_tt + 15.0 * n_stops
    round_trip = 2.0 * one_way
    headway = float(b["dispatching_headway"][0])
    return {
        "one_way_run_sec": one_way,
        "round_trip_sec": round_trip,
        "headway_sec": headway,
        "suggested_fleet_size": max(2, round(round_trip / headway)),
    }


# --------------------------------------------------------------------------- #
# --preview : OLD placeholder vs NEW real (the target your integration must hit)
# --------------------------------------------------------------------------- #
def preview(direction: str) -> None:
    direction = direction.lower()
    C.banner(f"DAY 6 — PREVIEW target numbers  ({direction.upper()})")
    try:
        new = _load_bundle(direction)
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    # OLD = the shipped placeholder DataLoader (this import is torch-free).
    from setup.ttc_route_29_data.dataloader import DataLoader
    old = DataLoader(direction)

    old_hw = tuple(old.dispatching_headway)
    new_hw = tuple(new["dispatching_headway"])
    old_sp = old.get_spacing()                       # tt_mean * 20 km/h  (GAP 2 placeholder)
    new_sp = {str(k): float(v) for k, v in new["spacing"].items()}   # real haversine

    print(f"\n  {'quantity':<26}{'OLD (placeholder)':>20}{'NEW (your data)':>18}")
    print(f"  {'nodes':<26}{len(old.node_ids):>20}{len(new['node_ids']):>18}")
    print(f"  {'headway mean (s)':<26}{old_hw[0]:>20.0f}{new_hw[0]:>18.1f}")
    print(f"  {'headway std (s)':<26}{old_hw[1]:>20.0f}{new_hw[1]:>18.1f}")
    print(f"  {'total spacing (km)':<26}{sum(old_sp.values())/1000:>20.2f}"
          f"{sum(new_sp.values())/1000:>18.2f}")

    C.banner("What your Day-6 integration must achieve")
    print("  Make the TTC env report the NEW column above instead of the OLD one, i.e.:")
    print("   * GAP 1 — headway: read it from your bundle, not the hardcoded (300, 60).")
    print("   * GAP 2 — spacing: read real haversine spacing, not tt_mean * 20 km/h.")
    print("  HOW you do that is your call — see docs/06_integrate_into_simulator.md for the")
    print("  contract and design options. Then validate with:  --check")

    fs = suggest_fleet_size(direction)
    C.banner("Suggested config_quick_run.yaml (paste + tune)")
    print(f"  # one-way run ~{fs['one_way_run_sec']:.0f}s, round-trip ~{fs['round_trip_sec']:.0f}s")
    print(f"  env_name: 'ttc_route_29_{direction}_real'   # the env YOU register")
    print("  has_schedule: no")
    print(f"  fleet_size: {fs['suggested_fleet_size']}      # ~ round_trip / headway")
    print("  episode_duration: 21600")
    print("  metric_names: ['headway_std', 'hold_time', 'pax_boarding_rejection']")
    print("  running_agent: 'Do_Nothing'   # baseline; then compare a holding agent")


# --------------------------------------------------------------------------- #
# --check : validate YOUR registered env (does not build it for you)
# --------------------------------------------------------------------------- #
def check(direction: str, env_name: str | None = None) -> None:
    direction = direction.lower()
    env_name = env_name or f"ttc_route_29_{direction}_real"
    C.banner(f"DAY 6 — CHECK your env  ('{env_name}')")
    os.chdir(C.CODE_ROOT)   # setup/chengdu loads data by a cwd-relative path

    try:
        from setup.blueprint import Blueprint
    except Exception as e:
        print(f"  Could not import setup.blueprint: {e}")
        return
    try:
        bp = Blueprint(env_name)
        if getattr(bp, "route_schema", None) is None:
            raise KeyError(env_name)
    except Exception:
        print(f"  '{env_name}' is not registered (or failed to build).")
        print("  Build your real env under setup/ and add a branch for it in")
        print("  setup/blueprint.py, then re-run this check. See docs/06.")
        return

    route = list(bp.route_schema.route_details_by_id.values())[0]
    lengths = [g.length for g in bp.network.link_geometry_info.values()]
    n_nodes = len(route.visit_seq_stops) + 2

    ok = True
    print(f"\n  nodes                 : {n_nodes}")
    print(f"  schedule_headway (s)  : {route.schedule_headway:.1f}", end="")
    if route.schedule_headway > PLACEHOLDER_HEADWAY[0]:
        print("   [ok: real, > 300 placeholder]")
    else:
        print("   [FAIL: still the 300 s placeholder — GAP 1 not fixed]"); ok = False
    print(f"  min link spacing (m)  : {min(lengths):.1f}", end="")
    if min(lengths) > 0 and max(lengths) < 20000:
        print("   [ok: real positive spacing]")
    else:
        print("   [FAIL: spacing looks like a placeholder/garbage — GAP 2]"); ok = False

    # cross-check against your derived bundle, if present
    try:
        b = _load_bundle(direction)
        if abs(route.schedule_headway - float(b["dispatching_headway"][0])) < 1.0:
            print("  headway matches your derived bundle   [ok]")
        else:
            print(f"  NOTE: env headway {route.schedule_headway:.1f}s != bundle "
                  f"{float(b['dispatching_headway'][0]):.1f}s (using different data?)")
    except FileNotFoundError:
        pass

    C.banner("PASS" if ok else "NOT YET — fix the FAILs above")
    if ok:
        print("  Your env is driven by real data. Run it:  --run-sim  (or quick_run.py)")


# --------------------------------------------------------------------------- #
# --run-sim : short DoNothing episode on YOUR registered env
# --------------------------------------------------------------------------- #
def run_sim(direction: str, env_name: str | None = None, duration: int = 3600) -> None:
    direction = direction.lower()
    env_name = env_name or f"ttc_route_29_{direction}_real"
    C.banner(f"DAY 6 — RUN-SIM (DoNothing, {duration}s)  '{env_name}'")
    os.chdir(C.CODE_ROOT)
    try:
        from setup.blueprint import Blueprint
        from simulator.simulator import Simulator
        from agent.do_nothing import DoNothing
    except Exception as e:
        print(f"  Could not import the simulator/agent stack: {e}")
        print("  Run this on the machine where quick_run.py works.")
        return
    try:
        bp = Blueprint(env_name)
        if getattr(bp, "route_schema", None) is None:
            raise KeyError(env_name)
    except Exception:
        print(f"  '{env_name}' is not registered — do your Day-6 integration first (docs/06).")
        return

    rc = {"episode_num": 1, "episode_duration": duration,
          "fleet_size": suggest_fleet_size(direction)["suggested_fleet_size"],
          "hold_start_time": 0, "hold_end_time": duration, "has_schedule": False,
          "metric_names": ["headway_std", "hold_time"], "warm_up": False,
          "env_name": env_name}
    agent = DoNothing({"agent_name": "Do_Nothing"}, bp)
    sim = Simulator(bp, agent, rc)
    act: dict = {}
    for t in range(duration):
        snap = sim.step(t, act)
        act = agent.calculate_hold_time(snap)
    metrics, _ = sim.get_metrics()
    print(f"  buses dispatched: {len(sim.total_buses)}")
    print(f"  metrics: {metrics}")
    print("  -> now compare this headway_std against a holding agent "
          "(Forward_Headway_Control / Simple_Control_Nonlinear) via quick_run.py.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 6 — companion tool for wiring your data into the sim")
    ap.add_argument("--check", action="store_true", help="validate YOUR registered real env")
    ap.add_argument("--run-sim", action="store_true", help="run a short DoNothing episode on your env")
    ap.add_argument("--direction", default="north", choices=["north", "south"])
    ap.add_argument("--env-name", default=None, help="override the env_name to check/run")
    args = ap.parse_args()

    if args.check:
        check(args.direction, args.env_name)
    elif args.run_sim:
        run_sim(args.direction, args.env_name)
    else:
        for d in ("north", "south"):
            preview(d)
        C.banner("DAY 6 — this tool does NOT wire the env for you")
        print("Do the integration yourself (docs/06_integrate_into_simulator.md), then:")
        print("  python scripts/day6_integrate.py --check      # validate your env")
        print("  python scripts/day6_integrate.py --run-sim    # run it (or quick_run.py)")


if __name__ == "__main__":
    main()
