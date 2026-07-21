# TTC Route 29 — Real-Data-Grounded Simulation Environment

## What you are building (and why)

The bus-holding RL simulator in this repo currently runs on an environment that was
**fit to a Chengdu route from thin static inputs** — a handful of guessed numbers stand
in for real geometry, speeds, headways, and demand. Your job is to replace those guesses
with an environment **derived from real Toronto Transit Commission (TTC) Route 29
"DUFFERIN" data**: November 2023 AVL (GPS) traces and APC (automatic passenger counts),
plus a GTFS static feed. When you are done, the simulator will hold buses on a route whose
stop spacing, travel-time distributions, dispatching headway, per-stop demand, and
origin–destination flows all come from measured data instead of placeholders — a
reproducible pipeline that any future student can re-run.

## The one-sentence mission

> **Derive geometry, speed/travel-time, dispatching headway, per-stop demand, and an
> origin–destination matrix from `raw_data/` into one reproducible pipeline, then wire it
> into `setup/` and run the simulator on the real Route-29 env (Day 6 capstone).**

---

## Folder map

```
student_project/
├── README.md                     ← you are here: the entry point
├── requirements.txt              ← Python deps (already satisfied by the repo env)
├── docs/
│   ├── 01_architecture.md        ← how the simulator/setup layer is wired together
│   ├── 02_data_dictionary.md     ← every raw file + column you will touch
│   ├── 03_data_to_env_mapping.md ← which raw signal becomes which env number (+ the 5 gaps)
│   ├── 04_week_plan.md           ← day-by-day plan (Day 1 → Day 6 capstone)
│   ├── 05_deliverables_checklist.md ← what "done" looks like; how you are graded
│   └── 06_integrate_into_simulator.md ← Day 6: wire your data into setup/ + run the sim
├── scripts/
│   ├── common.py                 ← shared paths/loaders/helpers — IMPORT, don't duplicate
│   ├── day1_explore.py           ← sanity-check the raw data, print shapes/coverage
│   ├── day2_geometry.py          ← derive_geometry(): stop order + haversine spacing
│   ├── day3_travel_time.py       ← derive_travel_time(): stop-to-stop tt mean/std/cv
│   ├── day4_headway.py           ← derive_headway(): real dispatching headway
│   ├── day5_demand_od.py         ← derive_demand_od(): per-stop lambda + IPF OD matrix
│   ├── build_env_data.py         ← assemble everything into DataLoader-shaped outputs
│   └── day6_integrate.py         ← Day 6 companion: preview target / --check your env / --run-sim
├── tests/
│   ├── conftest.py               ← shared pytest fixtures
│   ├── test_raw_data.py          ← guards the raw inputs (files present, columns exist)
│   ├── test_env_smoke.py         ← Blueprint("ttc_route_29_north") still constructs
│   ├── test_derived_data.py      ← your derived numbers are sane (non-neg, upper-tri OD, …)
│   └── test_integration.py       ← your bundle is ready to drive the sim (torch-free)
├── outputs/                      ← EVERYTHING you generate lands here
│
│   ─── bundled so this folder is self-contained (copies of the repo pieces) ───
├── raw_data/                     ← the AVL / APC / GTFS source data
├── setup/                        ← the env building blocks (Network, Route_Schema, DataLoader…)
├── simulator/                    ← the bus-operation simulator
├── agent/                        ← the control agents (DoNothing, holding controllers, RL)
├── config.py  quick_run.py       ← run entry points
└── config_quick_run.yaml         ← run configuration (you edit this on Day 6)
```

> **Self-contained:** `student_project/` bundles its own `raw_data/`, `setup/`, `simulator/`,
> `agent/`, and run entry points, so it works on its own. If you were handed *only* this folder,
> `cd` into it and drop the `student_project/` prefix from the commands below (e.g.
> `python scripts/day1_explore.py`). The scripts auto-detect whether they are standalone or
> embedded in the larger repo.

---

## Setup

### Prerequisites

* **Python 3.10+**
* The main lab machine already has everything. Two tiers of dependencies, both pinned in
  `requirements.txt`:
  * **Days 1–5 + tests:** `pandas, numpy, scipy, matplotlib, networkx, pytest, tabulate`.
  * **Day 6 (running the simulator):** also `pyyaml, torch, joblib` — pulled in by the bundled
    `agent/` + `config.py`. `torch` is heavy/platform-specific; use the build matching your
    machine if it is not already installed.

If you are in a fresh virtualenv, install them with:

```bash
pip install -r student_project/requirements.txt
```

### Sanity command (run this first)

All commands in this project are run **from the repo root** `/home/jiahao/Documents/busoperation`.

```bash
python student_project/scripts/common.py
```

This prints where the project expects its data (`REPO_ROOT`, `RAW`, `SETUP_DATA`, `OUT`)
and confirms every raw file is present. You should see an `[ok]` next to each of
`ttc_apc_clean_data.csv`, `ttc_apc_data.csv`, `ttc_avl_seg_clean_data.csv`, and the GTFS
`.txt` files. If any shows `MISSING`, stop and fix your `raw_data/` before going further.

---

## Do this first

Work through the docs and scripts **in order**. Each doc tells you what to read, what to
run, and what "done" looks like.

1. **Read [`docs/01_architecture.md`](docs/01_architecture.md)** — understand how
   `Blueprint → Network + Route_Schema → DataLoader` fits together, so you know what your
   numbers plug into.
2. **Read [`docs/02_data_dictionary.md`](docs/02_data_dictionary.md)** — learn every raw
   file and column before you trust a single value.
3. **Read [`docs/03_data_to_env_mapping.md`](docs/03_data_to_env_mapping.md)** — the map
   from raw signal → env parameter, and the **5 gaps** (below) you are here to fix.
4. **Follow [`docs/04_week_plan.md`](docs/04_week_plan.md)** day by day. In order, run:

   ```bash
   python student_project/scripts/day1_explore.py       # Day 1: explore & sanity-check
   python student_project/scripts/day2_geometry.py      # Day 2: spacing from Lat/Lon
   python student_project/scripts/day3_travel_time.py   # Day 3: travel-time distributions
   python student_project/scripts/day4_headway.py       # Day 4: real dispatching headway
   python student_project/scripts/day5_demand_od.py     # Day 5: lambda + OD matrix (IPF)
   ```

   Each script runs standalone in under ~60s, prints a readable report (and, where it
   replaces a placeholder, prints the **OLD** value next to the **NEW** measured value),
   and writes its outputs into `student_project/outputs/`.
5. **Assemble the environment data:**

   ```bash
   python student_project/scripts/build_env_data.py
   ```

   This calls every `derive_*()` for NORTH and SOUTH and writes a DataLoader-compatible
   bundle to `student_project/outputs/derived/<north|south>/`
   (`data_<dir>.pickle`, `summary_<dir>.json`, `stops_<dir>.csv`, `links_<dir>.csv`,
   `od_rate_matrix_<dir>.csv`). It is **non-destructive** — it never writes into `setup/` —
   and it prints a comparison of your regenerated summary against the reference in
   `setup/ttc_route_29_data/`.
6. **Run the tests:**

   ```bash
   pytest student_project/tests -v
   ```

   Green means your raw inputs are intact, the existing environment still builds, and your
   derived numbers are sane.
7. **Capstone — Day 6: build & run the real env (open-ended, your design).** Follow
   [`docs/06_integrate_into_simulator.md`](docs/06_integrate_into_simulator.md). There is **no
   ready-made answer in `setup/`** — you design how to feed the sim your data; the tool only
   previews the target and checks your work:

   ```bash
   python student_project/scripts/day6_integrate.py            # preview the target (writes nothing)
   # ... you build + register your real env under setup/ (your design) ...
   python student_project/scripts/day6_integrate.py --check    # validate YOUR env -> PASS
   python quick_run.py --plot                                  # run it; then a holding agent
   ```

   This is the payoff: the simulator running on your measured Route-29 data, and a
   baseline-vs-holding-agent comparison of `headway_std` / bunching.
8. **Check yourself against [`docs/05_deliverables_checklist.md`](docs/05_deliverables_checklist.md).**

---

## The 5 gaps you are fixing

These placeholders are the whole point of the project. See
[`docs/03_data_to_env_mapping.md`](docs/03_data_to_env_mapping.md) for full detail and the
exact lines in `setup/ttc_route_29_data/dataloader.py`.

* **GAP 1 — Headway is hardcoded.** `dataloader.py`'s `dispatching_headway` returns a
  guessed `(300, 60)` seconds. The **real** NORTH terminal-departure headway is
  mean ≈ 536 s, median ≈ 499 s — roughly **double** the placeholder. Fix it in Day 4.
* **GAP 2 — Geometry is faked from time.** `get_spacing()` estimates spacing as
  `tt_mean × 20 km/h` instead of using real distance. The raw APC table carries per-stop
  Lat/Lon at ~100% coverage, so true spacing = cumulative **haversine** along the stop
  sequence. Fix it in Day 2.
* **GAP 3 — No reproducible pipeline.** The pickles/CSVs in `setup/ttc_route_29_data/`
  exist, but **no script regenerates them** from `raw_data/`. Building a clean, documented
  ETL is the core deliverable (`day*` scripts + `build_env_data.py`).
* **GAP 4 — Provenance mismatch.** The GTFS feed is a **2020** snapshot while the AVL/APC
  are **November 2023**; only ~72% of 2023 stop_ids appear in the 2020 GTFS. Prefer APC
  Lat/Lon for geometry; treat GTFS as a cross-check / source of shape polylines only.
* **GAP 5 — Demand is coupled to the wrong headway.** Per-stop `lambda` was computed as
  `avg_boarding_per_trip / 300` (the placeholder headway). Fixing GAP 1 changes every
  lambda — Day 5 makes this coupling explicit.

---

## How your work reaches the simulator

```
raw_data/                     (APC + AVL + GTFS, Nov 2023)
    │
    ▼
day1_explore  →  day2_geometry  →  day3_travel_time  →  day4_headway  →  day5_demand_od
    │                 │                   │                   │                │
    │           spacing (m)         tt mean/std/cv     headway (mean,std)  lambda + OD
    └──────────────────────────────┬────────────────────────────────────────────┘
                                    ▼
                       scripts/build_env_data.py  (derive_all → outputs/derived/)
                                    │
                                    ▼   Day 6 (YOUR design — docs/06):
        a DataLoader that reads your bundle, with the headway + spacing placeholders removed
                                    │
                                    ▼
        your real Network + Route_Schema  (imitating setup/ttc_route_29.py)
                                    │
                                    ▼
              setup/blueprint.py  →  Blueprint("ttc_route_29_north_real")   (you register it)
                                    │
                                    ▼
                       simulator/simulator.py  →  Simulator(blueprint, agent, run_config)
```

Days 1–5 produce the derived data at the top; it lands in
`student_project/outputs/derived/` so nothing in the working simulator can break. **Day 6**
(the capstone) is *your* open-ended task: wire that bundle into `setup/` so the sim reports real
headway and spacing instead of the `(300, 60)` / `20 km/h` placeholders, then run it.
`day6_integrate.py` previews the target and `--check`s your work; it does **not** do the wiring
for you. See [`docs/06_integrate_into_simulator.md`](docs/06_integrate_into_simulator.md).

Happy deriving — start with [`docs/01_architecture.md`](docs/01_architecture.md).
