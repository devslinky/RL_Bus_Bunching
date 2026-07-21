# 03 — Data → Environment Mapping (the crux document)

**Read this after [01_architecture.md](01_architecture.md) and [02_data_dictionary.md](02_data_dictionary.md).**
This is the single most important doc in the handoff. It answers one question for
every knob the simulator reads:

> *Where does this number come from, what is it set to **today**, and what should
> **you** replace it with using the real TTC data?*

The simulator does **not** read raw CSVs. It reads a small set of Python objects
(`Blueprint → Network + Route_Schema`) that in turn read a `DataLoader`. So the whole
game is: **make `DataLoader` return real, reproducible numbers instead of the
placeholders that are baked in today.** Every placeholder below is real, still in the
code, and is one of your deliverables to fix.

**How the numbers flow (top to bottom):**

```
raw_data/*.csv  (TTC Nov-2023 AVL + APC, + 2020 GTFS)
      │   ← YOU build this ETL (it does not exist yet — GAP 3)
      ▼
student_project/scripts/dayN_*.py  →  build_env_data.py
      │   writes student_project/outputs/derived/<dir>/data_<dir>.pickle + summary_<dir>.json
      ▼
setup/ttc_route_29_data/dataloader.py   DataLoader(direction)   ← the seam you eventually repoint
      ▼
setup/ttc_route_29.py   TTC_Route_29_Network + TTC_Route_29_Route_Schema
      ▼
setup/blueprint.py   Blueprint("ttc_route_29_north")   ← the object the Simulator consumes
```

---

## 1. The mapping table

Columns:

- **Env parameter** — the physical quantity.
- **DataLoader property / consumer field** — where it enters the code and who reads it (`file:line`).
- **Real data source** — which raw table + which script derives it.
- **Current value in repo** — what it is *right now*. **PLACEHOLDER** = fake, fix it.
- **What you should derive** — your deliverable.

| Env parameter | DataLoader property → consumer (`file:line`) | Real data source | Current value in repo | What you should derive |
|---|---|---|---|---|
| **Ordered stop list** (nodes) | `node_ids` → `ttc_route_29.py:41`, `:126` | `ttc_apc_clean_data.csv` StopID/StopSeq via `C.stop_sequence()` | Reference env: 48 nodes N / 50 S. | `day2_geometry.derive_geometry()["node_ids"]` — reproduced from APC yields **49 N / 50 S** (all unique StopIDs; +1 over the reference N, which dropped/merged one low-traffic stop). The 49 is *more faithful* — keep it and note the difference. |
| **Terminals** (start/end) | `terminal_start_id`, `terminal_end_id` → `ttc_route_29.py:130-131` | APC first/last StopSeq | Real: N `11991`→`2108`, S `2107`→`11992`. **OK.** | Confirm from `stop_sequence` head/tail; carry through `build_env_data`. |
| **Inter-stop spacing** (link length, meters) | `get_spacing()` → `ttc_route_29.py:43, 54-57, 84-88` (feeds `LinkGeometry.length` + node x-coords) | `ttc_apc_data.csv` per-stop **Lat/Lon** → cumulative **haversine** | **PLACEHOLDER** `dataloader.py:105-120`: `spacing = tt_mean × (20 km/h)`. Pure guess. **GAP 2** | `day2_geometry.derive_geometry()["spacing"]`: true haversine spacing keyed by downstream stop id. |
| **Link travel time** (mean, std, cv) | `link_time_info` `{tail: {loc, scale}}` → `ttc_route_29.py:42, 90-94` (feeds `LinkDistribution`) | `ttc_apc_clean_data.csv` consecutive **StopArrivalTime** within a TripID | Real numbers exist (`links_north.csv`, mean_tt ≈ 50 s) **but not regenerable** (GAP 3). | `day3_travel_time.derive_travel_time()` → per-link `tt_mean/tt_std/tt_cv/tt_count`. |
| **Dispatching / schedule headway** (mean, std sec) | `dispatching_headway` → `ttc_route_29.py:135` → `_define_schedule_headway()` `:178-180` | `ttc_apc_clean_data.csv` consecutive terminal departures (StopSeq==1, StopDepartureTime) | **PLACEHOLDER** `dataloader.py:79-88`: hardcoded `return 300, 60`. **~half the true value. GAP 1** | `day4_headway.derive_headway()` → overall ≈ `(536, 295)` s + by-period. |
| **Per-stop demand λ** (pax/sec) | `stop_pax_arrival_rate` → `ttc_route_29.py:127` (Route_Schema boarding source) | `ttc_apc_clean_data.csv` Boarding ÷ headway | Real column exists (`stops_north.csv:lambda_pax_sec`) **but divided by the fake 300 s headway.** **GAP 5** | `day5_demand_od.derive_demand_od()["lambda"]`: boardings-per-trip ÷ **real** headway. |
| **OD demand matrix** (pax/sec) | `od_rate_table` `{o:{d:rate}}` → `ttc_route_29.py:149` `_define_od_table()` | `ttc_apc_clean_data.csv` Boarding (row margins) + Alighting (col margins), IPF | Real IPF matrix exists (`od_rate_matrix_north.csv`) **but not regenerable & coupled to fake headway.** | `day5_demand_od.derive_demand_od()["od_rate"]`: IPF, upper-triangular, ÷ real headway. |
| **Boarding service rate** (pax/sec at door) | `_define_boarding_rate()` → `ttc_route_29.py:198-206` | `ttc_apc_clean_data.csv` AvgDwell vs Boarding (could be fit) | **PLACEHOLDER** `ttc_route_29.py:205`: flat `1/4.0` (4 s/pax) for every stop. | *Optional stretch:* fit from AvgDwell. Not a required gap — note it and move on. |
| **Berths per stop** | `StopNodeGeometry(x, y, berth_num)` → `ttc_route_29.py:45, 73` | (not in data) | **ASSUMPTION** `berth_num = 2`. | Leave as-is; document the assumption. |
| **Bus capacity** | `_define_bus_capacity()` (optional) | GTFS/vehicle roster | Default (not overridden). | Leave as-is unless you have vehicle data. |
| **Reproducible ETL** (the pipeline itself) | *n/a — no script exists* | `raw_data/` → `setup/ttc_route_29_data/` | **MISSING.** The pickles exist; the code that made them is lost. **GAP 3** | `build_env_data.derive_all()` + `build_env_data.main()` — the whole `student_project/scripts/` pipeline. |
| **Geometry provenance** (which lat/lon is truth) | (affects spacing + any GTFS join) | 2023 APC Lat/Lon **vs** 2020 GTFS stops.txt | 2020 GTFS only matches ~72% of 2023 stops. **GAP 4** | Prefer APC Lat/Lon; use GTFS shapes only as a cross-check, never as the 2023 stop truth. |

> **Reading tip.** Open a reference target next to this table so you can see the shape
> you are aiming for: `setup/ttc_route_29_data/stops_north.csv`,
> `links_north.csv`, `od_rate_matrix_north.csv`, `summary_north.json`. Your regenerated
> outputs in `student_project/outputs/derived/north/` must mirror those.

---

## 2. THE GAP LIST — your deliverables, in detail

Each gap has: the **exact anchor**, the **placeholder value**, the **real value / evidence**,
and **which script fixes it**. Fixing all five is the project.

---

### GAP 1 — Dispatching headway is hardcoded to ~half the true value

**Anchor:** `setup/ttc_route_29_data/dataloader.py`, lines **79–88**.

```python
@property
def dispatching_headway(self) -> Tuple[int, float]:
    # Based on TTC schedule analysis:
    # - Peak: ~3 min (180 sec)
    # - Off-peak: ~8-10 min (480-600 sec)
    return 300, 60          # ← PLACEHOLDER: guessed, not measured
```

Read at `ttc_route_29.py:135` and surfaced to the simulator via
`_define_schedule_headway()` (`ttc_route_29.py:178-180`). So this one fake number sets
how often buses are dispatched in every experiment.

**Placeholder:** `(mean, std) = (300, 60)` seconds. The docstring even admits it is a
guess.

**Real value (measured from APC terminal departures, NORTH):**

| Slice | mean (s) | median (s) | std (s) |
|---|---|---|---|
| **Overall** | **~536** | **~499** | **~295** |
| AM peak | ~556 | — | — |
| Midday | ~500 | — | — |
| PM peak | ~548 | — | — |
| Late evening | ~689 | — | — |

The placeholder `300 s` is **roughly half** the true `~536 s`. Halving the headway
roughly **doubles** dispatch frequency, and (via GAP 5) inflates per-stop λ, so this
error propagates everywhere.

**How to measure it:** consecutive terminal departures — rows where `StopSeq == 1`, take
`StopDepartureTime`, sort within each service date, diff, keep positive diffs in
`(30, 3600)` s.

**Fixed by:** **Day 4** → `day4_headway.derive_headway(direction)` returns
`{"overall": (mean, std), "by_period": {period: (mean, std, n)}}`. See
[04_week_plan.md](04_week_plan.md) (Day 4) and the script `scripts/day4_headway.py`.
`build_env_data` then writes this into `dispatching_headway`.

> **STUDENT TODO:** decide whether the env should use one overall headway or switch
> headway by `PeriodID`. Overall is the safe default; per-period is a clean extension.

---

### GAP 2 — Stop spacing is faked from travel time, not real geometry

**Anchor:** `setup/ttc_route_29_data/dataloader.py`, lines **105–120**.

```python
def get_spacing(self) -> Dict[str, float]:
    avg_speed_mps = 20 * 1000 / 3600          # ← PLACEHOLDER: assumed 20 km/h everywhere
    spacing = {}
    for link in self.link_info:
        spacing[str(link['to_stop_id'])] = link['tt_mean'] * avg_speed_mps
    return spacing
```

Read at `ttc_route_29.py:43` and used both to place nodes on the x-axis and to set
`LinkGeometry.length` (`ttc_route_29.py:84-88`, `:102-104`).

**Placeholder:** `spacing = tt_mean × (20 km/h)`. This is circular — it derives *distance*
from *time* by assuming a constant speed, so geometry is just a rescaled copy of travel
time. A slow (congested) link looks *longer*, which is physically wrong.

**Real value / evidence:** `ttc_apc_data.csv` carries per-stop **Lat/Lon** with
**~100% coverage** on Route 29. True inter-stop spacing = **cumulative haversine** along
the canonical stop sequence:

```
spacing[tail] = haversine(lat[head], lon[head], lat[tail], lon[tail])   # meters
```

`common.py` already gives you `C.stop_latlon(apc_raw, direction)` and
`C.haversine(...)`. GTFS `shapes.txt`/`shape_dist_traveled` is a *cross-check* only — see
GAP 4.

**Fixed by:** **Day 2** → `day2_geometry.derive_geometry(direction)["spacing"]`
(dict keyed by the **downstream** stop id, matching the `DataLoader` convention; the first
terminal has no entry). See [04_week_plan.md](04_week_plan.md) (Day 2) and
`scripts/day2_geometry.py`.

> **STUDENT TODO:** haversine is straight-line ("as the crow flies") distance; real
> streets curve. Note the assumption. If you want, sanity-check against
> `shapes.txt shape_dist_traveled` and report the ratio — but keep haversine as the
> default because the shape feed is a 2020 snapshot (GAP 4).

---

### GAP 3 — There is no reproducible ETL from `raw_data/` to the DataLoader inputs

**Anchor:** the *absence* of any script. The derived files
(`setup/ttc_route_29_data/data_north.pickle`, `stops_north.csv`, `links_north.csv`,
`od_rate_matrix_north.csv`, `summary_north.json`, …) **exist**, but **no script in the
repo regenerates them from `raw_data/`.** The pipeline that made them is lost and
undocumented.

**Why it matters:** un-reproducible data is not science. You cannot fix GAP 1 or GAP 2 in
a durable way if you cannot rebuild the pickle the simulator reads. If a reviewer asks
"how did you get 50.1 s mean travel time?", today the answer is "we don't know."

**What you build:** the whole `student_project/scripts/` pipeline, ending in
`build_env_data.py`:

- `day1_explore.py` — sanity/coverage checks on the raw tables.
- `day2_geometry.derive_geometry()` — GAP 2.
- `day3_travel_time.derive_travel_time()` — link travel-time distributions.
- `day4_headway.derive_headway()` — GAP 1.
- `day5_demand_od.derive_demand_od()` — λ + OD (GAP 5).
- `build_env_data.derive_all(direction)` — assembles a **DataLoader-compatible** dict and
  `build_env_data.main()` writes `student_project/outputs/derived/<north|south>/`:
  `data_<dir>.pickle`, `summary_<dir>.json`, `stops_<dir>.csv`, `links_<dir>.csv`,
  `od_rate_matrix_<dir>.csv`.

**Non-destructive rule:** the *derivation* pipeline (Days 1–5) **never** writes into `setup/`
or `raw_data/` — everything lands in `student_project/outputs/`. Wiring the real simulator to
your regenerated data is the **Day 6 capstone**, and it is *your* open-ended design task — see
[06_integrate_into_simulator.md](06_integrate_into_simulator.md).

**Fixed by:** the whole week — Days 1–5 build the reproducible dataset (`build_env_data.py`),
and Day 6 runs it in the simulator. See [04_week_plan.md](04_week_plan.md).

---

### GAP 4 — Provenance mismatch: 2020 GTFS vs 2023 AVL/APC

**Anchor:** `raw_data/calendar.txt` spans **2020-10-11 … 2020-11-21**, while
`ttc_apc_*` and `ttc_avl_*` are **November 2023**. Only **~72%** of the 2023 APC Route-29
`StopID`s appear in the 2020 GTFS `stops.txt`.

**Why it matters:** it is tempting to grab "official" geometry from GTFS `stops.txt` /
`shapes.txt`. But that feed is **three years stale** and does not describe the 2023 stop
set. Joining 2023 trips to 2020 stops silently drops ~28% of stops and mislocates others.

**The rule (repeat it in every script that touches GTFS):**

- **Geometry → use APC Lat/Lon** (2023, ~100% coverage). This is ground truth for 2023.
- **GTFS → cross-check only.** Use `shapes.txt` for a shape polyline / a spacing
  sanity ratio; never as the authoritative 2023 stop list.
- When you *do* join to GTFS, **report the match rate** so the reader knows how much you
  trusted it.

**Addressed in:** **Day 2** (choose APC over GTFS for geometry) and threaded through
**Day 1, Day 3, Day 5** wherever GTFS is mentioned. There is no single "fix" — it is a
discipline: prefer the 2023 data, and be explicit whenever you reach for the 2020 feed.

---

### GAP 5 — Per-stop λ and the OD matrix are coupled to the (wrong) headway

**Anchor:** `setup/ttc_route_29_data/stops_north.csv`, column `lambda_pax_sec`. It was
computed as

```
lambda_pax_sec = avg_boarding_per_trip / 300          # 300 = the GAP-1 placeholder headway
```

Verify it yourself: row 1 (`StopID 11991`) has `avg_boarding_per_trip = 0.35576…` and
`lambda_pax_sec = 0.0011858…` — and `0.35576 / 300 = 0.0011858`. The `300` is the fake
headway from GAP 1.

**Why it matters:** λ is a *rate* (pax **per second**). To convert boardings-per-trip into
a per-second rate you divide by the headway (the seconds between buses). If the headway is
wrong, **λ is wrong by the same factor** — here ~1.8× too high (300 vs ~536). The OD rate
matrix inherits the same error because it, too, is scaled to a per-second rate.

**The coupling, stated plainly:**

```
lambda_i        = (mean boardings per trip at stop i) / headway_sec
od_rate[i][j]   = (IPF count share of i→j) / trips_observed / headway_sec
```

So **fixing GAP 1 forces you to recompute GAP 5.** That is why
`day5_demand_od.derive_demand_od(direction, headway_sec=None)` takes the headway as an
argument: if `headway_sec is None` it imports `day4_headway` and uses the *real* overall
mean, making the dependency explicit and correct by default.

**Real method for OD (IPF):**

- Row margins `O_i` = total boardings at stop i; column margins `D_j` = total alightings at
  stop j (rescale so `Σ O == Σ D`).
- Seed an N×N matrix with a **feasibility mask**: entry `(i, j)` allowed only if origin i is
  strictly **upstream** of dest j in the canonical stop order (strictly upper-triangular),
  else 0.
- Iterate ~50×: scale rows to `O`, scale columns to `D`, until margins converge.
- Convert the resulting **count** matrix to a **rate** matrix by dividing each origin row by
  the number of trips observed and then by `headway_sec` (equivalently, distribute `lambda_i`
  across destinations by the IPF row proportions). Result must be non-negative and strictly
  upper-triangular.

**Fixed by:** **Day 5** → `day5_demand_od.derive_demand_od()` returns
`{"lambda", "od_rate", "boardings", "alightings"}`. See [04_week_plan.md](04_week_plan.md)
(Day 5) and `scripts/day5_demand_od.py`.

> **STUDENT TODO:** if you implement per-period headway (GAP 1 TODO), decide whether λ and
> OD should also become per-period. Keep one overall headway as the default so the pipeline
> always runs.

---

## 3. Which script fixes which gap (quick cross-reference)

| Gap | Symptom | Script / function | Deliverable output |
|---|---|---|---|
| **GAP 1** headway | hardcoded `(300, 60)` | `day4_headway.derive_headway()` | `dispatching_headway` = `(≈536, ≈295)` + by-period |
| **GAP 2** spacing | `tt_mean × 20 km/h` | `day2_geometry.derive_geometry()` | haversine `spacing` dict + `cum_dist_m` |
| **GAP 3** no ETL | pickles unregenerable | `build_env_data.derive_all()` / `.main()` | `outputs/derived/<dir>/data_<dir>.pickle` + `summary` |
| **GAP 4** provenance | 2020 GTFS vs 2023 data | *all scripts* (prefer APC, report match rate) | documented choice + match-rate print |
| **GAP 5** λ/OD coupling | λ = boardings/300 | `day5_demand_od.derive_demand_od()` | λ + IPF OD scaled by **real** headway |
| (link tt) | not regenerable | `day3_travel_time.derive_travel_time()` | `link_time_info` `{tail:{loc,scale}}` |

(Travel time is not a "gap" — the numbers are plausible — but it must be regenerated as
part of GAP 3 so the whole pickle is reproducible.)

---

## 4. What "done" looks like

You are done with the mapping when:

1. `python student_project/scripts/build_env_data.py` runs clean from the repo root and
   writes `student_project/outputs/derived/{north,south}/`.
2. The printed comparison shows your regenerated **headway ≈ 536 s** (not 300) and your
   **spacing from haversine** (not `tt×20 km/h`), side-by-side with the old placeholders.
3. Your `summary_<dir>.json` matches the reference on terminal ids and link count, is
   *more faithful* on stop count (**49 N** — every unique APC StopID — vs the reference's
   48; SOUTH matches at 50), and *improves* the things that were fake (headway, spacing,
   and the λ/OD that depend on headway).
4. Every place you used GTFS, you printed the 2020↔2023 match rate (GAP 4 discipline).

Then comes the **Day 6 capstone** (your open-ended design task): install a real-data env under
`setup/` whose loader reads your bundle with the `(300, 60)` and `20 km/h` placeholders removed,
and run the actual simulator on it. `day6_integrate.py` previews the target and `--check`s your
work — see [06_integrate_into_simulator.md](06_integrate_into_simulator.md).

**Next:** [04_week_plan.md](04_week_plan.md) turns this gap list into a day-by-day plan.
