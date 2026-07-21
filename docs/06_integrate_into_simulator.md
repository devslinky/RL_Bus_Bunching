# 06 — Integrate your derived data into the simulator (the capstone)

Days 1–5 gave you a DataLoader-compatible bundle in
`student_project/outputs/derived/{north,south}/`. **This day is the payoff, and it is
deliberately open-ended: you design and build the integration yourself.** The point is to
make the real `simulator/` run on *your* numbers — the real ~536 s headway (not 300 s) and
the real haversine spacing (not `tt × 20 km/h`) — then run a bus-holding agent on it and
measure the effect on bunching.

There is **no ready-made answer in `setup/`** and this doc does not hand you one. It gives you
the contract, the design options, and a companion tool to check your work. How you wire it is
your call.

> Read [`01_architecture.md`](01_architecture.md) first if the `DataLoader → Network /
> Route_Schema → Blueprint → Simulator` chain is not fresh.

---

## 1. The one fact that makes this tractable

`simulator/simulator.py` is **generic**: it consumes a `Blueprint`, which is a `Network` + a
`Route_Schema`, both fed by a `DataLoader`. So you do **not** touch the simulator's dynamics.
You make a `DataLoader` hand the env *your* numbers. Everything downstream just works.

The shipped TTC env reads these off its DataLoader (see
`setup/ttc_route_29_data/dataloader.py` and how `setup/ttc_route_29.py` uses them):

| DataLoader member | feeds | real? in the shipped loader |
|---|---|---|
| `node_ids`, `terminal_start_id`, `terminal_end_id` | stop topology | real |
| `stop_pax_arrival_rate`, `od_rate_table` | demand | real (but λ tied to 300 s) |
| `link_time_info` (`{tail: {loc, scale}}`) | travel-time distributions | real |
| **`dispatching_headway`** | schedule headway | **PLACEHOLDER — hardcoded `(300, 60)`** |
| **`get_spacing()`** | link lengths / geometry | **PLACEHOLDER — `tt_mean × 20 km/h`** |

**Your job:** make the env report real values for the two **bold** rows (and keep the rest),
sourced from your Day 2–5 bundle. Your bundle already contains everything needed — including
the two missing pieces — under these keys in `data_<dir>.pickle`:

```
node_ids, terminal_start_id, terminal_end_id, stop_pax_arrival_rate,
link_time_info, od_rate_table, spacing, dispatching_headway, link_info, stop_info
```

`test_integration.py` verifies your bundle carries all of these with sane values, so you can
trust the data before you start wiring.

---

## 2. See the target (non-destructive)

```bash
python scripts/day6_integrate.py            # (or student_project/scripts/... if embedded)
```

`--preview` prints, per direction, the **OLD placeholder vs NEW (your data)** numbers your
integration must make the env report (e.g. NORTH headway `300 → 536.1 s`, spacing totals), and
a suggested `config_quick_run.yaml` block with a `fleet_size` derived from your own travel
times and headway. It writes nothing.

---

## 3. Design it (your choice)

Pick an approach. Both are legitimate; **Option B is recommended** because it keeps the shipped
reference env intact and is trivially reversible.

**Option A — edit the shipped loader in place.**
Back it up first (`cp -r setup/ttc_route_29_data setup/ttc_route_29_data.bak`), drop your
regenerated `data_<dir>.pickle` in, and change the two placeholder methods in
`setup/ttc_route_29_data/dataloader.py` to read `dispatching_headway` and the spacing from
`self.data` instead of returning `(300, 60)` / `tt_mean × 20 km/h`. Simple, but it mutates the
env everyone else compares against.

**Option B — add a new `ttc_route_29_<dir>_real` env (recommended).**
Leave the shipped env untouched and add a parallel one. The pieces you will create — figure out
the details by reading the shipped files they mirror:

1. **A data location** for your bundle under `setup/` (e.g. a new `setup/ttc_route_29_data_real/`
   holding your `data_<dir>.pickle` + `summary_<dir>.json`). It is *your* derived output, not a
   provided answer — copy it there yourself.
2. **A DataLoader** that reads that bundle and returns real values for the two placeholder
   members. Study `setup/ttc_route_29_data/dataloader.py`: you only need to change what
   `dispatching_headway` and `get_spacing()` return (subclassing it is one clean way).
3. **A Network + Route_Schema** for the env. Study `setup/ttc_route_29.py` — notice its network
   already calls `get_spacing()` and its schema already reads `dispatching_headway`, so if your
   loader returns real values, these need very little new code (subclass and inject your loader).
4. **An env-name registration.** Study how `setup/blueprint.py` maps `'ttc_route_29_north'` to
   its Network/Schema, and add a `'ttc_route_29_<dir>_real'` branch the same way.
5. **Run config.** Point `config_quick_run.yaml` at your env_name (use the block `--preview`
   printed).

> Name your env **`ttc_route_29_<direction>_real`** so the check/run tools below find it (or
> pass `--env-name`).

---

## 4. Check your work

```bash
python scripts/day6_integrate.py --check                 # NORTH (default)
python scripts/day6_integrate.py --check --direction south
```

`--check` builds `Blueprint('ttc_route_29_<dir>_real')` (your registration) and asserts the env
now reports a real headway (> 300 s) and real positive spacing, and that the headway matches your
bundle. It tells you PASS or exactly which GAP is still unfixed. It does not build the env for
you — it validates the one you built.

---

## 5. Run it and do the experiment

```bash
python scripts/day6_integrate.py --run-sim               # short DoNothing episode, prints metrics
# or the standard entry point (edit config_quick_run.yaml first):
python quick_run.py --plot                               # writes a time-space diagram
```

Then the actual point of the project — compare **the same real env** under a do-nothing baseline
vs a holding controller and report the change in bunching:

```bash
# config_quick_run.yaml: running_agent: 'Do_Nothing'          -> python quick_run.py --plot
# config_quick_run.yaml: running_agent: 'Forward_Headway_Control' (set has_schedule: yes)
#                                                              -> python quick_run.py --plot
```

Report the change in **`headway_std`** and eyeball the two time-space diagrams (bunched platoons
vs evenly spaced). A good holding agent visibly reduces headway variance — on a *real-data*
Route 29 instead of the thin Chengdu fit.

---

## 6. Optional deeper `setup/` / `simulator/` changes (stretch)

- **Finite bus capacity** from APC loads (`setup/ttc_route_29.py`, `_define_bus_capacity`) so
  `board_truncation` / `rejection_events` bite. Derive a number from observed peak `Depload`.
- **Data-driven boarding rate** from `AvgDwell` (`_define_boarding_rate`, currently flat `1/4.0`).
- **Time-of-day env** variants (per-period headway + OD from Days 4–5).
- **A crowding / headway-CV metric** in `simulator/tracer.py` — the one place a `simulator/`
  code change is justified — then list it in `metric_names`.

---

## 7. Definition of done

- [ ] `--preview` shows your NEW numbers (real headway ~536 s N / ~524 s S, real spacing).
- [ ] You built a real-data env under `setup/` (your design) and registered it in
      `setup/blueprint.py`.
- [ ] `python scripts/day6_integrate.py --check` prints **PASS** (headway from data, real spacing).
- [ ] `quick_run.py` (or `--run-sim`) runs an episode on your real env and reports metrics.
- [ ] You ran **baseline vs holding-agent** and reported the change in `headway_std`, with a
      time-space diagram for each.
- [ ] Your `REPORT.md` has a "Simulation" section: what you changed in `setup/`, the run config,
      and the bunching result.

**Reversible (Option B):** delete the files you added under `setup/` and remove your
`blueprint.py` branch to restore the shipped state.
