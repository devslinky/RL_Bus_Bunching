# 05 — Deliverables Checklist (Definition of Done)

This is your self-check rubric **and** your mentor's grading sheet. Tick every box.
A deliverable is "done" only when the concrete pass-criterion next to it is true —
not when the script merely runs without crashing.

**How to use this doc**

- Work day-by-day. Each section maps to one script and its outputs.
- Copy-paste the commands. They assume you run **from the repo root**
  `/home/jiahao/Documents/busoperation`.
- When every box in a section is ticked, commit and ask your mentor to grade that section.
- The five **GAPS** (the whole reason this project exists) are called out with a 🎯.
  Read [`03_data_to_env_mapping.md`](03_data_to_env_mapping.md) for what each gap is and
  [`04_week_plan.md`](04_week_plan.md) for the day-by-day plan.

**One-line grading summary the mentor fills in at the end:**

> Reproducible ETL from `raw_data/` → DataLoader-compatible env data, with the three
> placeholders (300 s headway, 20 km/h spacing, no pipeline) replaced by real measured
> values, a passing test suite, sanity plots, and a written report. ✅ / ❌

---

## 0. Environment & setup (do this first)

- [ ] Repo cloned; you can `cd /home/jiahao/Documents/busoperation`.
- [ ] Dependencies installed: `pip install -r student_project/requirements.txt`.
- [ ] Raw data present:
      `ls raw_data/` shows `ttc_apc_clean_data.csv`, `ttc_apc_data.csv`,
      `ttc_avl_seg_clean_data.csv`, and the GTFS `.txt` files.
- [ ] You have **read** these docs, in order:
      [`01_architecture.md`](01_architecture.md),
      [`02_data_dictionary.md`](02_data_dictionary.md),
      [`03_data_to_env_mapping.md`](03_data_to_env_mapping.md),
      [`04_week_plan.md`](04_week_plan.md).
- [ ] You have **skimmed** `scripts/common.py` and understand its public API
      (`C.load_apc_clean()`, `C.route_dir()`, `C.stop_sequence()`, `C.stop_latlon()`,
      `C.haversine()`, `C.save_csv/save_json/savefig`, `C.OUT`, `C.banner`).
      You **import** it; you never re-implement it.

**Pass criterion:** you can explain, in two sentences, what
`Blueprint("ttc_route_29_north")` builds today (48 nodes = 46 stops + 2 terminals,
47 links, OD table, `terminal_start 11991 → terminal_end 2108`, headway `(300, 60)`).

---

## 1. Day 1 — Explore the raw data (`scripts/day1_explore.py`)

Run:

```bash
python student_project/scripts/day1_explore.py
```

- [ ] Script runs clean (exit 0, no traceback), in well under 60 s.
- [ ] Prints, per direction, the row counts and the **49 NORTH StopIDs** in
      `StopSeq` order (1..49), terminal → terminal.
- [ ] Confirms `Route == 29` always; `RouteDirection ∈ {NORTH, SOUTH}`;
      lists the `Branch` values (`DLWI`, `29Dcon`, `DLPRcon`) and `PeriodID` values.
- [ ] Reports APC raw **Lat/Lon coverage** on Route-29 NORTH (should be ~100 %) —
      this is the evidence that geometry from APC is trustworthy (🎯 GAP 2).
- [ ] Flags the **provenance mismatch** (🎯 GAP 4): only ~72 % of 2023 APC stop_ids
      appear in the 2020 GTFS `stops.txt`; date ranges printed side by side.
- [ ] `outputs/` gains the Day-1 artifacts: `day1_summary.json` (which now includes
      `apc_latlon_coverage`, `gtfs_stop_overlap`, and the GTFS calendar span) plus the
      four PNGs `day1_boardings_per_stop_north.png`, `day1_load_profile_north.png`,
      `day1_headway_hist_north.png`, `day1_avl_kph_hist_north.png`.

**Pass criterion:** the printout convinces the reader that APC is the source of truth for
geometry/demand and GTFS is only a cross-check.

---

## 2. Day 2 — Geometry / spacing (`scripts/day2_geometry.py`) 🎯 GAP 2

Run:

```bash
python student_project/scripts/day2_geometry.py
```

- [ ] `derive_geometry("NORTH")` is **importable** and returns a dict with keys
      `node_ids`, `spacing`, `stops` (per the interface contract in
      [`03_data_to_env_mapping.md`](03_data_to_env_mapping.md)).
- [ ] `node_ids` is a `list[str]` in canonical terminal→terminal order and **equals**
      `C.stop_sequence(...)` order.
- [ ] `spacing` is `{tail_stop_id(str): meters(float)}`, keyed by the **downstream** stop
      (DataLoader convention); the first terminal stop has **no** entry.
- [ ] Spacing is computed as **cumulative haversine** along the stop sequence from the
      real APC Lat/Lon — **NOT** `tt_mean * 20 km/h`.
- [ ] The script **prints the OLD vs NEW** value for a few links:
      the 20 km/h estimate next to the real haversine spacing (they should differ clearly).
- [ ] Total route length prints and is **physically sane** for Route 29 DUFFERIN
      (order of ~15–20 km end to end; no negative or absurd link lengths).
- [ ] `outputs/` gains `day2_geometry_north.csv` (with `Lat, Lon, cum_dist_m`) and
      `day2_route_map_north.png` (a Lon-vs-Lat stop map) so you can eyeball the route shape.

**Pass criterion:** every consecutive link has a positive real-distance spacing derived
from GPS coordinates, and the printout shows the 20 km/h placeholder being replaced.

---

## 3. Day 3 — Travel time distributions (`scripts/day3_travel_time.py`)

Run:

```bash
python student_project/scripts/day3_travel_time.py
```

- [ ] `derive_travel_time("NORTH")` is **importable** and returns a DataFrame with
      **exactly** columns `[from_stop_id, to_stop_id, tt_mean, tt_std, tt_cv, tt_count]`
      (stop ids as `str`, times in **seconds**).
- [ ] Travel times are per-trip, stop-to-stop diffs of consecutive `StopArrivalTime`
      **within the same `TripID`**, walked along the canonical stop order.
- [ ] Absurd values filtered (keep ~`5..1200` s). `tt_cv == tt_std / tt_mean`.
- [ ] Links with `tt_count == 0` fall back to a **stated** default
      (`tt_mean 50`, `tt_cv 0.3`) and the script says which links used the fallback.
- [ ] Aggregate stats print and are near the reference:
      **mean_tt ≈ 50 s**, **mean_tt_cv ≈ 0.55**, `num_links_with_data ≈ 46`
      (compare to `setup/ttc_route_29_data/summary_north.json`).
- [ ] `outputs/` gains `day3_travel_time_north.csv` and `day3_travel_time_north.png`
      (a per-link mean±std travel-time plot along the route).

**Pass criterion:** a `link_time_info`-ready table exists whose aggregate mean/CV match
the reference within a sensible tolerance, with fallbacks explicitly reported.

---

## 4. Day 4 — Dispatching headway (`scripts/day4_headway.py`) 🎯 GAP 1

Run:

```bash
python student_project/scripts/day4_headway.py
```

- [ ] `derive_headway("NORTH")` is **importable** and returns
      `{"overall": (mean_sec, std_sec), "by_period": {period: (mean_sec, std_sec, n)}}`.
- [ ] Headway computed from consecutive **terminal departures**: rows with `StopSeq == 1`,
      `StopDepartureTime` sorted **within each service date**, positive diffs filtered to
      `(30, 3600)` s.
- [ ] Measured NORTH overall is in the right ballpark:
      **mean ≈ 536 s, median ≈ 499 s, std ≈ 295 s** (i.e. ~500 s, not 300 s).
- [ ] `by_period` prints and roughly matches:
      AM ≈ 556 s, Midday ≈ 500 s, PM ≈ 548 s, Late evening ≈ 689 s.
- [ ] The script **prints OLD vs NEW explicitly**: the hardcoded placeholder
      `(300, 60)` next to the measured `(≈536, ≈295)`, noting the placeholder is
      **roughly half** the true value.
- [ ] `outputs/` gains `day4_headway_north.csv` (per-period rows) and a **histogram**
      of terminal-departure headways.

**Pass criterion:** the real headway (~500 s) is measured, documented, and shown
replacing the `(300, 60)` placeholder — the single most important number in the project.

---

## 5. Day 5 — Demand λ and OD matrix (`scripts/day5_demand_od.py`) 🎯 GAP 5

Run:

```bash
python student_project/scripts/day5_demand_od.py
```

- [ ] `derive_demand_od("NORTH", headway_sec=None)` is **importable** and returns
      `{"lambda", "od_rate", "boardings", "alightings"}`.
- [ ] `lambda[stop_id]` = (mean boardings per trip at that stop) / `headway_sec`.
      When `headway_sec is None`, the script **imports Day 4** and uses its overall mean.
- [ ] The **headway coupling (🎯 GAP 5) is explicit in a comment**: fixing the headway
      (Day 4) changes every λ, because the reference `lambda_pax_sec` was
      `avg_boarding_per_trip / 300`. With ~500 s headway, λ shrinks by ~40 %.
- [ ] The script prints **λ computed at 300 s vs at the real headway** for a few stops
      so the impact is visible.
- [ ] `od_rate` is produced by **IPF**:
      - row margins `O_i` = total boardings, column margins `D_j` = total alightings,
        rescaled so `sum(O) == sum(D)`;
      - seed masked so entry `(i, j)` is allowed **only if origin i is strictly upstream
        of dest j** in canonical order;
      - ~50 iterations (or until margins converge);
      - counts → rates by dividing by trips observed and by `headway_sec`.
- [ ] **OD invariants hold** (assert them in the script):
      - matrix is **non-negative** everywhere,
      - matrix is **strictly upper-triangular** in stop order (no same-stop, no upstream flow),
      - **each origin row sum ≈ that stop's λ** (this is the key consistency check tying
        Day 5's OD back to Day 5's λ — print the max abs difference, it should be ~0).
- [ ] `outputs/` gains `day5_lambda_north.csv` (λ per stop), `day5_od_rate_matrix_north.csv`,
      and `day5_od_heatmap_north.png`.

**Pass criterion:** λ is coupled to the real headway (not 300 s), and the OD matrix is
non-negative, strictly downstream, and its per-origin row sums equal the per-stop λ.

---

## 6. Assemble the env data (`scripts/build_env_data.py`)

Run:

```bash
python student_project/scripts/build_env_data.py
```

- [ ] `derive_all("NORTH")` assembles a **DataLoader-compatible** dict with keys:
      `node_ids`, `terminal_start_id`, `terminal_end_id`, `stop_pax_arrival_rate` (== λ),
      `link_time_info` (`{tail_stop_id: {loc: tt_mean, scale: tt_std}}`),
      `od_rate_table` (nested `origin→dest→rate`, downstream pairs only),
      `spacing`, `dispatching_headway` (`(mean, std)` from Day 4),
      plus `link_info` and `stop_info` lists.
- [ ] `main()` runs `derive_all` for **both NORTH and SOUTH** and writes into
      `student_project/outputs/derived/<north|south>/`:
      - [ ] `data_<dir>.pickle`
      - [ ] `summary_<dir>.json`
      - [ ] `stops_<dir>.csv`
      - [ ] `links_<dir>.csv`
      - [ ] `od_rate_matrix_<dir>.csv`
- [ ] The build is **NON-DESTRUCTIVE**: nothing under `setup/` or `raw_data/` is touched.
      (Confirm with `git status setup/ raw_data/` → no changes from the build.)
- [ ] It **prints a comparison** of regenerated summary vs the reference
      `setup/ttc_route_29_data/summary_<dir>.json`: num stops, mean tt, and headway
      **old (300) vs new (~500)**.
- [ ] It ends by printing a **Day-6 pointer**: that you next wire this bundle into the
      simulator yourself (section 8), removing the `(300, 60)` and 20 km/h placeholders.

**Pass criterion:** `outputs/derived/north/` and `outputs/derived/south/` each contain all
five files; the summary comparison prints; `setup/` is untouched.

---

## 7. Tests (`pytest`)

Run:

```bash
python -m pytest student_project/tests/ -q
```

- [ ] **`test_raw_data.py`** passes — raw CSVs load, expected columns exist, Route is 29,
      directions/branches/periods are as documented.
- [ ] **`test_derived_data.py`** passes — the `derive_*` functions return the contracted
      shapes; OD is non-negative + strictly upper-triangular; row sums ≈ λ; headway is in a
      sane range (well above 300 s).
- [ ] **`test_env_smoke.py`** passes — the assembled dict / regenerated data can stand in
      for the DataLoader shape without error.
- [ ] **`test_integration.py`** passes — the Day-6 real-data env builds with headway and
      spacing from your bundle (torch-free; no simulator/agent import needed).
- [ ] Whole suite green: `pytest -q` reports **0 failures, 0 errors**.

**Pass criterion:** `pytest student_project/tests/ -q` exits 0.

---

## 8. Day 6 — Integrate into the simulator (capstone) 🎯

**Open-ended: you design and build the integration.** No answer is provided in `setup/`.
Full guide: [`06_integrate_into_simulator.md`](06_integrate_into_simulator.md).

```bash
python scripts/day6_integrate.py            # preview the target numbers (writes nothing)
# ... you build + register your real env (your design) ...
python scripts/day6_integrate.py --check    # validate YOUR env  -> PASS
```

- [ ] `day6_integrate.py` (preview) shows OLD-vs-NEW with **real headway ~536 s N / ~524 s S**
      and real spacing — the target your integration must hit.
- [ ] You built a real-data env under `setup/` (your design) that sources `dispatching_headway`
      and spacing from your bundle instead of the `(300, 60)` / 20 km/h placeholders.
- [ ] You registered a new `env_name` (e.g. `ttc_route_29_north_real`) in `setup/blueprint.py`
      and pointed `config_quick_run.yaml` at it (sane `fleet_size`, `episode_duration`,
      `metric_names`).
- [ ] `python scripts/day6_integrate.py --check` prints **PASS** (headway from data > 300,
      real positive spacing, headway matches your bundle).
- [ ] `python quick_run.py --plot` (or `day6_integrate.py --run-sim`) runs an episode on your
      real env and reports metrics + a time-space diagram.
- [ ] **The experiment:** you ran `Do_Nothing` vs a holding agent
      (`Forward_Headway_Control` / `Simple_Control_Nonlinear`) on the real env and reported the
      change in **`headway_std`**, with a time-space diagram for each.

**Pass criterion:** the simulator runs on your derived Route-29 data, `--check` passes, and you
can show — with a number and a picture — how a holding agent changes bunching on the real env.

---

## 9. Written report (`outputs/REPORT.md` — you write this)

A short (1–3 page) report living at `student_project/outputs/REPORT.md`.
For **each** derived quantity, give three things: **method**, **result**, **caveats**.

- [ ] **Geometry / spacing** — method (cumulative haversine over APC Lat/Lon);
      result (total length, example link spacings); caveat (replaces 20 km/h estimate;
      GPS jitter; GTFS shapes only cross-checked, not used, 🎯 GAP 4).
- [ ] **Travel time** — method (per-trip consecutive `StopArrivalTime` diffs, filter 5..1200 s,
      normal `loc/scale`); result (mean ≈ 50 s, CV ≈ 0.55); caveat (fallback links; branch mixing).
- [ ] **Dispatching headway** — method (terminal `StopSeq==1` departure diffs, per date, 30..3600 s);
      result (≈536 s mean, per-period spread); caveat (**replaces (300, 60)** 🎯 GAP 1;
      branches share a terminal).
- [ ] **Demand λ** — method (mean boardings/trip ÷ headway); result (per-stop λ pax/s);
      caveat (**coupled to headway**, 🎯 GAP 5 — 300 s vs 500 s changes every λ).
- [ ] **OD matrix** — method (IPF with upstream-only mask, boardings=row margin,
      alightings=col margin); result (upper-triangular rate matrix); caveat (IPF assumes a
      separable/gravity-like structure; no true OD ground truth in APC).
- [ ] **Simulation (Day 6)** — what you changed in `setup/` to wire the real env, the run
      config used, and the baseline-vs-holding-agent bunching result (`headway_std` +
      time-space diagrams).
- [ ] **Provenance** — one paragraph stating why APC (Nov 2023) beats GTFS (2020 snapshot)
      for 2023 geometry/demand (🎯 GAP 4).
- [ ] Report explicitly names all **five GAPS** and states which are now closed and which
      remain (e.g. branch handling, distribution choice).

**Pass criterion:** a reader who never saw the code understands what each number is, how it
was measured, and what not to trust.

---

## 10. Quality bar (graded across the whole submission)

### Reproducibility
- [ ] Deleting `outputs/derived/` and re-running `build_env_data.py` yields **byte-identical**
      (or numerically identical) CSVs/JSON — no randomness leaks in. Verify:

  ```bash
  python student_project/scripts/build_env_data.py
  md5sum student_project/outputs/derived/north/*.csv > /tmp/run1.md5
  rm -rf student_project/outputs/derived
  python student_project/scripts/build_env_data.py
  md5sum -c /tmp/run1.md5      # every line must say: OK
  ```
- [ ] Any IPF / sampling step uses a **fixed seed** or is fully deterministic.
- [ ] Every script runs **standalone from repo root** with a plain
      `python student_project/scripts/<file>.py` — no manual setup, no `cd` gymnastics,
      each under ~60 s.

### Documentation
- [ ] Every derived quantity's **method + assumptions** are written down (in `REPORT.md`
      and/or module docstrings).
- [ ] Every script has a module docstring, uses `C.banner` for sections, and prints
      **OLD-placeholder-vs-NEW-real** wherever it replaces a placeholder.
- [ ] `# STUDENT TODO:` hooks mark the real decision points (branch handling, distribution
      choice, period segmentation) — and the script still runs correctly without any TODO done.

### Validation (sanity plots — all in `outputs/`)
- [ ] Stop-location scatter (route shape looks like Dufferin, not spaghetti).
- [ ] Link travel-time histogram (no absurd tail after filtering).
- [ ] Headway histogram (mass around ~500 s, not ~300 s).
- [ ] OD heatmap (visibly upper-triangular, zeros below the diagonal).
- [ ] At least one **cross-check** plot (e.g. AVL `KPH`-implied travel time vs APC-derived
      travel time, or GTFS `shape_dist_traveled` vs haversine spacing) to show the numbers
      agree with an independent source.

**Pass criterion:** rerun is identical, methods are written down, and the five plots exist
and look right.

---

## 11. Stretch / going further (bonus — attempt after the core is green)

None of these are required to pass, but each is a real, useful extension.
(Wiring the real env in and measuring bunching are **not** here — they graduated into the
required **Day 6** capstone, section 8.)

- [ ] **Time-of-day scenarios.** Emit separate env data per `PeriodID` (AM peak / Midday /
      PM peak / Late evening) — different headway **and** different λ per scenario — so the
      simulator can be run under peak vs off-peak conditions.
- [ ] **Branch handling.** Split or weight by `Branch` (`DLWI`, `29Dcon`, `DLPRcon`) instead
      of pooling all trips; decide how short-turn branches affect the stop set and OD.
- [ ] **Distribution fitting beyond normal.** Fit lognormal / gamma to link travel times and
      dwell (right-skewed by nature); compare AIC/BIC to the normal `loc/scale` and report
      which better matches the tails.
- [ ] **AVL-based travel time.** Reconstruct stop-to-stop times from `ttc_avl_seg_clean_data.csv`
      `KPH` + spacing (ignore the corrupt `Distance` column) and reconcile with the APC-derived
      times.
- [ ] **Finite bus capacity / boarding rate / crowding metric.** Deeper `setup/` (and one
      optional `simulator/tracer.py`) changes — see
      [`06_integrate_into_simulator.md`](06_integrate_into_simulator.md) §4.

---

## Final sign-off

- [ ] Sections 0–10 all ticked (section 11 is bonus).
- [ ] `pytest student_project/tests/ -q` is green.
- [ ] `outputs/derived/{north,south}/` complete; your Day-6 integration is the **only** change
      under `setup/` (and, if you took the additive route, is reversible).
- [ ] `day6_integrate.py --check` passes; real env runs in `quick_run.py`;
      baseline-vs-holding-agent bunching reported.
- [ ] `outputs/REPORT.md` written; all five 🎯 GAPS addressed.
- [ ] Mentor has graded and signed:  **Grade: ____   Date: ____**
