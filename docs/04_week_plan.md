# 04 — Week Plan (5 derivation days + 1 integration capstone)

Welcome aboard. This is your day-by-day playbook for turning raw TTC Route 29 data
into a real-data-grounded simulation environment. Each day builds on the last: by the end of
**Day 5** you have a complete, reproducible dataset in `student_project/outputs/derived/`,
and **Day 6** (the capstone) wires it into `setup/` and runs the real simulator on it.

**Read this alongside:**
- [`README.md`](../README.md) — project overview and setup
- [`docs/01_architecture.md`](01_architecture.md) — how the simulator/Blueprint fits together
- [`docs/02_data_dictionary.md`](02_data_dictionary.md) — every raw column, explained
- [`docs/03_data_to_env_mapping.md`](03_data_to_env_mapping.md) — which raw field becomes which env number (and the 5 GAPs)
- [`docs/05_deliverables_checklist.md`](05_deliverables_checklist.md) — the tick-box list you sign off against

---

## The big picture (read once, then start)

The simulator is fed by a `Blueprint` object. For Route 29 that Blueprint pulls every
number from `setup/ttc_route_29_data/dataloader.py`. That file **works today**, but it
is riddled with placeholders that a previous developer left behind, and **nobody knows
how the derived data files were built** — the pipeline is lost.

Your job this week is to rebuild that pipeline from the raw data, cleanly and
reproducibly, and in doing so replace the placeholders with real measured values.
The five gaps you are fixing (all detailed in [`docs/03_data_to_env_mapping.md`](03_data_to_env_mapping.md)):

| Gap | What's wrong today | Day you fix it |
|-----|--------------------|----------------|
| **GAP 1 — Headway** | `dataloader.py` hardcodes dispatching headway `(300, 60)` s; real value is ~**500–540 s** | Day 4 |
| **GAP 2 — Geometry** | `get_spacing()` fakes distance as `tt_mean × 20 km/h`; real Lat/Lon exists | Day 2 |
| **GAP 3 — No pipeline** | The derived files exist but no script regenerates them | Day 5 (assembly) |
| **GAP 4 — Provenance** | GTFS is a 2020 snapshot; APC/AVL are Nov 2023 — only ~72% of stops match | Day 2 (why we trust APC) |
| **GAP 5 — Demand coupling** | `lambda` was computed with the *wrong* 300 s headway | Day 5 |

### Golden rules

1. **Everything is non-destructive.** Every script writes only into
   `student_project/outputs/`. You will **not** touch `setup/` or `raw_data/` until the
   explicit, opt-in "graduation" step at the very end of the week. You literally cannot
   break the working simulator by running these scripts.
2. **Run everything from the repo root:** `/home/jiahao/Documents/busoperation`.
   All commands below assume you are `cd`'d there.
3. **Keep runtime short.** Each day's script reads a few hundred-thousand rows and
   finishes in under ~60 s. If it hangs much longer, something is wrong — stop and read
   the error.

### One-time setup

```bash
cd /home/jiahao/Documents/busoperation

# Confirm the raw files are all present and paths resolve:
python student_project/scripts/common.py
```

You should see a path check and a list of raw files each marked `[ok]`. If any say
`MISSING`, fix that before Day 1 — nothing else will work.

---

## Day 1 — Orientation + Exploratory Data Analysis (EDA)

### Objective
Understand the shape of the raw data and how it maps to a bus-holding simulator, so the
rest of the week is informed rather than blind. No new numbers derived yet — this is
about building a correct mental model.

### Background to read
- [`docs/01_architecture.md`](01_architecture.md) — the `Blueprint → Network + Route_Schema` design.
- [`docs/02_data_dictionary.md`](02_data_dictionary.md) — read the whole thing; you'll refer back all week.
- [`docs/03_data_to_env_mapping.md`](03_data_to_env_mapping.md) — skim the 5 GAPs.
- In the repo, open and skim (don't run yet):
  - `setup/ttc_route_29_data/dataloader.py` — spot the `dispatching_headway` property (returns `300, 60`) and `get_spacing()` (uses `20 km/h`). These are the placeholders you'll kill.
  - `setup/ttc_route_29.py` — how the DataLoader numbers flow into `Network` + `Route_Schema`.
  - `setup/chengdu.py` — the *reference* pattern of a data-driven env; this is your template for "what good looks like."
  - `student_project/scripts/common.py` — the shared helper module. **Read the public API** (loaders, `route_dir`, `stop_sequence`, `stop_latlon`, `haversine`, the `save_*` helpers). You will import and reuse this all week; do not re-implement its logic.

### Tasks
1. Run the EDA script (below) and read every line it prints.
2. Open the plots it writes into `student_project/outputs/` and make sure you can explain
   each one to yourself.
3. In your own words, write a short **EDA summary** (~1 page, plain text or Markdown)
   answering:
   - How many unique stops does Route 29 NORTH have, and what are the two terminals?
   - What are the branches (`DLWI`, `29Dcon`, `DLPRcon`) and roughly how common is each?
   - What date range and day types (`MoTuWeThFr`, `Sa`, `Su`) does the APC data cover?
   - Eyeball: does boarding demand vary a lot stop-to-stop? Peak vs off-peak (`PeriodID`)?
   - One sentence each on where headway, spacing, travel time, and demand will *come from*.

### The script to run
```bash
python student_project/scripts/day1_explore.py
```

### Acceptance criteria — you are done when…
- The script ran clean (no traceback) and you've read its full printout.
- The plots exist in `student_project/outputs/` and you can explain each.
- You have a written EDA summary saved somewhere (e.g.
  `student_project/outputs/day1_eda_summary.md`) covering the five bullet questions.
- You can point at the two placeholder lines in `dataloader.py` from memory.

### Stretch goals
- Compare NORTH vs SOUTH: do they have the same stop count and terminals swapped?
- Load the raw AVL (`C.load_avl_seg()`) and plot the `KPH` distribution — note the pile-up
  of zeros (buses dwelling). You'll use this as a Day 3 cross-check.

---

## Day 2 — Geometry & Spacing (fixes GAP 2)

### Objective
Replace the fake `tt_mean × 20 km/h` spacing with **true inter-stop distances**,
computed as the cumulative haversine distance along the real stop Lat/Lon from the APC
data.

### Background to read
- [`docs/03_data_to_env_mapping.md`](03_data_to_env_mapping.md) — the **GAP 2** and **GAP 4** sections.
- `setup/ttc_route_29_data/dataloader.py` → `get_spacing()` (the `20 km/h` estimate you are replacing).
- In `common.py`: `stop_sequence()`, `stop_latlon()`, and `haversine()` — these do the heavy lifting.
- **Why APC Lat/Lon and not GTFS `shapes.txt`?** (GAP 4) The GTFS feed is a **2020** snapshot
  (its `calendar.txt` spans Oct–Nov 2020) while the APC data is **November 2023**. Only ~72%
  of 2023 stop_ids appear in the 2020 GTFS. The raw APC table (`ttc_apc_data.csv`) carries
  per-stop Lat/Lon with ~100% coverage, so it is the trustworthy geometry source. Treat GTFS
  as a cross-check / polyline source only.

### Tasks
1. Run the geometry script (below).
2. Inspect the spacing table it writes. Confirm the terminal (seq 1) has **no** spacing
   entry (spacing is keyed by the *downstream* stop, matching the DataLoader convention).
3. Compare **real spacing** vs the **fake `20 km/h` spacing** the script prints side by
   side. Note where they diverge most (long express-ish links vs short dense-stop links).
4. Look at the generated **route map** (stops plotted at their Lat/Lon, connected in
   sequence). Does it look like a real north–south corridor, or is a stop obviously
   mis-ordered? (If a stop jumps around, that's a branch artifact — see the TODO hook.)

### The script to run
```bash
python student_project/scripts/day2_geometry.py
```
The importable entry point is `day2_geometry.derive_geometry("NORTH")`, which returns
`{"node_ids", "spacing", "stops"}`. `build_env_data.py` will call this on Day 5, so make
sure it runs cleanly.

### Acceptance criteria — you are done when…
- A spacing table (per-link meters) exists in `student_project/outputs/`.
- A route-map plot exists and looks geographically sensible.
- You have a written comparison of real-haversine vs `20 km/h`-fake spacing, with the
  total route length under each method.
- `derive_geometry("NORTH")` and `derive_geometry("SOUTH")` both return without error.

### Stretch goals
- Overlay the GTFS `shapes.txt` polyline for route 29 (`route_id 61327`) on your route map
  and eyeball how well the 2023 stops sit on the 2020 shape. Quantify the ~72% match.
- Compute spacing two ways — straight haversine between stops vs distance *along* the GTFS
  shape — and report the difference (road curvature inflates the along-shape length).

---

## Day 3 — Travel Time & Speed

### Objective
Derive per-segment travel-time distributions — mean, std, and coefficient of variation
(CV) — from consecutive APC stop arrival times, and cross-check them against instantaneous
GPS speed (`KPH`) from the AVL data.

### Background to read
- [`docs/03_data_to_env_mapping.md`](03_data_to_env_mapping.md) — the travel-time / `LinkDistribution` mapping.
- `setup/config_dataclass.py` → `LinkDistribution(tt_mean, tt_cv, tt_type)` — the target shape.
- `setup/ttc_route_29_data/links_north.csv` — the **reference** output format
  (`link_seq, from_stop_id, to_stop_id, tt_mean, tt_std, tt_cv, tt_count`). Open it and note
  the mean tt ≈ **50 s** and mean CV ≈ **0.55** from `summary_north.json`.
- Columns you'll use: `TripID`, `StopSeq`, `StopArrivalTime` (from `ttc_apc_clean_data.csv`),
  and `KPH` (from `ttc_avl_seg_clean_data.csv`).

### Tasks
1. Run the travel-time script (below).
2. For each consecutive stop pair (in canonical order), the script computes per-trip
   travel time = difference of `StopArrivalTime` within the same `TripID`, filters absurd
   values (keep ~5–1200 s), and reports `tt_mean`, `tt_std`, `tt_cv`, `tt_count`.
3. Inspect the segment table. Which links are slowest / most variable (highest CV)? High
   CV links are exactly where bus bunching is born — note them.
4. **Cross-check with AVL speed:** convert your `tt_mean` and the Day-2 spacing into an
   implied speed (`spacing_m / tt_mean × 3.6` km/h) and compare against the AVL `KPH`
   median for the corridor (~13 km/h). They should be in the same ballpark; a wild
   mismatch flags a bad segment.
5. Write a short **note on CV**: why link travel-time CV matters for a bunching simulator
   (it's the noise that destabilizes headways), and whether CV varies by `PeriodID`.

### The script to run
```bash
python student_project/scripts/day3_travel_time.py
```
Importable entry point: `day3_travel_time.derive_travel_time("NORTH")` →
`DataFrame[from_stop_id, to_stop_id, tt_mean, tt_std, tt_cv, tt_count]`. Links with no
observations fall back to a sane default (`tt_mean=50`, `tt_cv=0.3`) — the script says so
when it happens.

### Acceptance criteria — you are done when…
- A segment travel-time table exists in `student_project/outputs/`, one row per link.
- Your mean travel time and mean CV are in the same ballpark as the reference
  `summary_north.json` (≈ 50 s, ≈ 0.55) — they won't match exactly, and that's fine;
  understand *why* (different filtering choices).
- At least one fitted-distribution plot exists (e.g. a histogram of travel times for one
  representative link with the fitted mean/std overlaid).
- You've written the CV note.

### Stretch goals
- Split travel time by `PeriodID` (AM peak vs Midday vs PM peak) for the 3–4 slowest links
  and quantify how much slower peak is. Decide (and write down) whether the env should be
  time-of-day dependent or a single all-day distribution — this is a real modelling choice.
- Fit a lognormal (`scipy.stats.lognorm`) to one link's travel times and compare against
  the normal assumption baked into `LinkDistribution`.

---

## Day 4 — Dispatching Headway (fixes GAP 1)

### Objective
Measure the **real** dispatching headway from terminal departures and prove that the
hardcoded `(300, 60)` s in `dataloader.py` is roughly **half** the true value.

### Background to read
- [`docs/03_data_to_env_mapping.md`](03_data_to_env_mapping.md) — the **GAP 1** section.
- `setup/ttc_route_29_data/dataloader.py` → the `dispatching_headway` property. Read the
  comment: it *guesses* peak/off-peak and returns `300, 60`. You are about to replace that
  guess with a measurement.
- `setup/route.py` → `_define_schedule_headway` — where this `(mean, std)` lands in the env.

### Tasks
1. Run the headway script (below).
2. It takes rows where `StopSeq == 1` (terminal departures), sorts `StopDepartureTime`
   within each service date, takes positive consecutive diffs, filters to (30 s, 3600 s),
   and reports the distribution **overall**, **by `PeriodID`**, and **by `DAYTYPE`**.
3. Read the printed **old-vs-new** comparison. Expect roughly:

   | Slice | Real headway (s) |
   |-------|------------------|
   | Overall mean | ~536 (median ~499, std ~295) |
   | AM peak | ~556 |
   | Midday | ~500 |
   | PM peak | ~548 |
   | Late evening | ~689 |

   The placeholder is **300 s**. Confirm your numbers land near these; if they're wildly
   off, check that you filtered to `StopSeq == 1` and one direction only.
4. Note the downstream consequence: **headway feeds demand.** Every per-stop `lambda` on
   Day 5 is `boardings_per_trip / headway`, so doubling the headway roughly halves the
   arrival-rate error. Write this coupling down (it's **GAP 5**).

### The script to run
```bash
python student_project/scripts/day4_headway.py
```
Importable entry point: `day4_headway.derive_headway("NORTH")` →
`{"overall": (mean, std), "by_period": {period: (mean, std, n)}}`.

### Acceptance criteria — you are done when…
- A headway summary table (overall + by period + by day type) exists in
  `student_project/outputs/`.
- A headway **histogram** exists, with the old `300 s` placeholder and your new mean marked
  on it (a vertical line each) so the ~2× gap is visually obvious.
- You can state, in one sentence, why the placeholder is dangerously wrong and what it
  breaks downstream.

### Stretch goals
- Plot headway by hour-of-day and see whether the AM/PM peaks actually tighten the headway
  (they should, in theory — check if the data agrees).
- Compare NORTH vs SOUTH headway. Are they dispatched symmetrically?

---

## Day 5 — Demand & OD + Full Assembly (fixes GAP 3 & GAP 5)

### Objective
Derive per-stop arrival rates (`lambda`) and an origin–destination (OD) rate matrix, then
**assemble everything from Days 2–4 into a DataLoader-compatible dataset** written to
`student_project/outputs/derived/`. This is the capstone: the reproducible pipeline whose
absence is GAP 3.

### Background to read
- [`docs/03_data_to_env_mapping.md`](03_data_to_env_mapping.md) — the **GAP 5** section and the OD/IPF description.
- [`docs/05_deliverables_checklist.md`](05_deliverables_checklist.md) — your final sign-off list.
- `setup/route.py` → `_define_od_table` and `_define_boarding_rate` — the targets.
- Reference outputs to mirror: `setup/ttc_route_29_data/stops_north.csv`,
  `od_rate_matrix_north.csv`, and `summary_north.json`. Open them so you know the shape you're
  reproducing.

### Tasks
1. **Demand & OD** — run the demand script:
   ```bash
   python student_project/scripts/day5_demand_od.py
   ```
   - Per-stop `lambda_i = (mean boardings per trip at i) / headway_sec`. If you pass no
     headway it uses Day 4's overall mean (the **correct** ~536 s, not 300 s). This is the
     explicit fix for **GAP 5** — the script prints lambda under both headways so you see
     the difference.
   - The OD matrix is built by **Iterative Proportional Fitting (IPF)**: row margins =
     total boardings per origin, column margins = total alightings per destination
     (rescaled so the totals match), seeded with an **upper-triangular feasibility mask**
     (a passenger can only alight *downstream* of where they boarded). Iterate rows↔columns
     ~50 times until margins converge, then convert counts to rates. The result must be
     non-negative and strictly upper-triangular.
   - Importable entry point: `day5_demand_od.derive_demand_od("NORTH")` →
     `{"lambda", "od_rate", "boardings", "alightings"}`.
2. **Assemble the environment** — run the builder:
   ```bash
   python student_project/scripts/build_env_data.py
   ```
   - It calls `build_env_data.derive_all("NORTH")` and `derive_all("SOUTH")`, stitching
     Day-2 geometry + Day-3 travel time + Day-4 headway + Day-5 demand into one dict shaped
     like the real `DataLoader` (keys: `node_ids`, `terminal_start_id`, `terminal_end_id`,
     `stop_pax_arrival_rate`, `link_time_info`, `od_rate_table`, `spacing`,
     `dispatching_headway`, plus `link_info`/`stop_info`).
   - It writes, per direction, into `student_project/outputs/derived/<north|south>/`:
     `data_<dir>.pickle`, `summary_<dir>.json`, `stops_<dir>.csv`, `links_<dir>.csv`,
     `od_rate_matrix_<dir>.csv`.
   - It **prints a comparison** of your regenerated `summary` vs the reference
     `setup/ttc_route_29_data/summary_north.json` — num stops, mean travel time, and
     **headway old-vs-new**. It also prints a **Day-6 pointer** (how to install the real env).
3. **Run the tests:**
   ```bash
   pytest student_project/tests/ -v
   ```
   `test_raw_data.py`, `test_derived_data.py`, `test_env_smoke.py`, and `test_integration.py`
   should all pass.

### Acceptance criteria — you are done when…
- `student_project/outputs/derived/north/` and `.../south/` each contain all five files.
- The printed comparison shows num_stops = 49 (NORTH — all unique APC stops; the reference
  shows 48), mean tt ≈ 50 s, and headway moved from 300 → ~500+ s.
- Your OD matrix is non-negative and strictly upper-triangular (the script asserts this).
- `pytest student_project/tests/ -v` is green.
- You've skimmed the Day-6 pointer and know `day6_integrate.py` comes next.

### Stretch goals
- Make the derivation **time-of-day aware**: feed the by-period headway (Day 4) and, as a
  TODO in `day5_demand_od.py`, per-period OD, instead of single all-day numbers.
- Load `outputs/derived/north/data_north.pickle` and confirm it carries the keys the
  DataLoader needs (see [`01_architecture.md`](01_architecture.md)).

> **Heads-up:** assembling the bundle is *not* the finish line. On **Day 6** you wire it into
> the real simulator and measure bunching — that is the capstone.

---

## Day 6 — Integrate into the simulator & measure bunching (the capstone)

### Objective
Make the real `simulator/` run on **your** derived numbers (real ~536 s headway, real
haversine spacing), then run a bus-holding agent on it and measure the effect on bunching.
This is where the analysis becomes an environment — the point of the whole project.

### Background to read
- [`06_integrate_into_simulator.md`](06_integrate_into_simulator.md) — the full step-by-step
  guide (read it before you touch anything).
- [`01_architecture.md`](01_architecture.md) — the `DataLoader → Network/Route_Schema →
  Blueprint → Simulator` contract you are completing.
- `config.py` (`build_simulation_elements`) and `quick_run.py` — how a run is configured and
  driven; `setup/blueprint.py` — where env names are registered.

> **This day is open-ended — you design the integration.** There is no ready-made answer in
> `setup/`; the tooling helps you preview the target and check your work, but the wiring is
> yours. [`06_integrate_into_simulator.md`](06_integrate_into_simulator.md) has the contract,
> the two placeholders, and the design options.

### Tasks
1. **See the target:** `python scripts/day6_integrate.py` — read the OLD-vs-NEW table
   (headway 300→~536 s, spacing) and the suggested run config. (Writes nothing.)
2. **Design your integration** (docs/06 §3). Make the TTC env source real values for the two
   placeholders (`dispatching_headway`, `get_spacing()`) from your Day 2–5 bundle. Recommended:
   add a `ttc_route_29_<dir>_real` env alongside the shipped one, imitating
   `setup/ttc_route_29_data/dataloader.py`, `setup/ttc_route_29.py`, and `setup/blueprint.py`.
3. **Register it** in `setup/blueprint.py` (a new `env_name` branch, same shape as the shipped
   `ttc_route_29_north` one) and **point** `config_quick_run.yaml` at it.
4. **Check it:** `python scripts/day6_integrate.py --check` → must print **PASS** (headway from
   data, real spacing).
5. **Run it:** `python quick_run.py --plot` (or `... day6_integrate.py --run-sim`).
6. **The experiment:** run the same env under `Do_Nothing` vs a holding controller
   (`Forward_Headway_Control` / `Simple_Control_Nonlinear`) and compare `headway_std` and the
   two time-space diagrams.

### The commands to run
```bash
python scripts/day6_integrate.py            # preview the target (safe, writes nothing)
# ... you build + register your real env (your design) ...
python scripts/day6_integrate.py --check    # validate YOUR env  -> PASS
python quick_run.py --plot                  # run it; then swap the agent and compare
```
(If `student_project/` is embedded in the repo, prefix the script path with `student_project/`.)

### Acceptance criteria — you are done when…
- `day6_integrate.py --check` prints **PASS** for the env you built (real headway > 300 s,
  real positive spacing, headway matching your bundle).
- `pytest tests/test_integration.py -q` is green (your bundle is contract-ready).
- `quick_run.py` runs an episode on your real env and reports metrics.
- You ran **baseline vs holding-agent** on the real env and reported the change in
  `headway_std`, with a time-space diagram for each.

### Stretch goals
- **Finite bus capacity** from APC loads (`setup/ttc_route_29.py`, `_define_bus_capacity`).
- **Data-driven boarding rate** from `AvgDwell` (`_define_boarding_rate`).
- **Time-of-day env** variants (per-period headway + OD).
- A **crowding / headway-CV metric** in `simulator/tracer.py` (the one place a `simulator/`
  code change is justified) — see [`06_integrate_into_simulator.md`](06_integrate_into_simulator.md) §4.

---

## End-of-week deliverable

By the end of the week you should be able to hand over:

1. **A written EDA summary** (Day 1) — the data explained in your own words.
2. **Real geometry** (Day 2) — haversine spacing table + route map + comparison to the
   fake `20 km/h` spacing. *(GAP 2 closed.)*
3. **Travel-time distributions** (Day 3) — per-segment `tt_mean/std/cv` table, fitted-dist
   plots, and a note on why CV drives bunching.
4. **Real headway** (Day 4) — overall/by-period/by-daytype table + histogram proving it's
   ~500 s, not 300 s. *(GAP 1 closed.)*
5. **Demand + OD + assembled env** (Day 5) — per-stop `lambda` (coupled to the *correct*
   headway, **GAP 5 closed**), an upper-triangular IPF OD matrix, and a complete
   DataLoader-compatible dataset in `student_project/outputs/derived/` with all tests
   green. *(GAP 3 closed: the pipeline exists and is reproducible.)*
6. **A running real-data simulation** (Day 6) — the `setup/` env wired to your bundle
   (`ttc_route_29_north_real`), a `quick_run.py` episode on it, and a **baseline vs
   holding-agent** bunching comparison (`headway_std` + time-space diagrams).

Cross off each item in [`docs/05_deliverables_checklist.md`](05_deliverables_checklist.md)
as you finish it.

### Reminder on what touches `setup/`

Days 1–5 write **only** to `student_project/outputs/` — the working simulator is safe no
matter what you run, so experiment fearlessly and re-run any day's script freely. **Day 6** is
the deliberate exception: *you* add a real-data env under `setup/` (your design — recommended
as new, additive files plus one new `blueprint.py` branch, so nothing existing is overwritten).
Keep it reversible: if you take the additive route, deleting your new files and branch restores
the shipped state (back up first if you instead edit the shipped loader in place).
