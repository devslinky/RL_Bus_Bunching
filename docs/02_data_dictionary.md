# 02 · Data Dictionary — everything in `raw_data/`

This is your field guide to the raw inputs. Read it once end-to-end, then keep it
open as a reference while you write the `dayN_*.py` scripts.

**What this covers:** every file under
`/home/jiahao/Documents/busoperation/raw_data/`, its shape, its columns (types,
units, meaning), the categorical value sets you will filter on, and — most
importantly — the **data-quality traps** that will bite you if you trust a column
blindly.

**Related reading (relative paths):**
- [`01_architecture.md`](01_architecture.md) — how the simulator consumes the derived numbers.
- [`03_data_to_env_mapping.md`](03_data_to_env_mapping.md) — which raw column becomes which env parameter, and the 5 GAPs you are fixing.
- [`04_week_plan.md`](04_week_plan.md) — the day-by-day plan that uses these files.

> **Golden rule:** the derivation scripts (Days 1–5) never write into `raw_data/` or
> `setup/` — load with the helpers in [`../scripts/common.py`](../scripts/common.py) and write
> results to `student_project/outputs/`. (Day 6 is the one, deliberate, *additive* exception:
> it installs the real env under `setup/`.)

---

## 0 · The observation window (read this first)

All the AVL/APC data covers **one month**:

| | value |
|---|---|
| First `StopArrivalTime` | `2023-11-01 04:48:07` |
| Last `StopArrivalTime`  | `2023-12-01 02:14:15` |
| Effective span | **2023-11-01 → 2023-12-01** (≈30 service days) |
| Route | always **29 DUFFERIN** |
| Directions | `NORTH`, `SOUTH` |

Everything you derive is an **aggregate over this month**. Counts (boardings,
alightings, trips) are *monthly totals or per-trip means*, not rates. To turn a
count into the **per-second rates** the simulator wants, you divide by a time
base:

```
lambda_i (pax/sec)  =  (mean boardings per trip at stop i)  /  dispatching_headway_sec
```

That headway divisor is the single most important number in the whole project,
and today it is a placeholder — see **GAP 1** in
[`03_data_to_env_mapping.md`](03_data_to_env_mapping.md).

---

## 1 · File inventory

| File | Rows (approx) | One row = | Primary job |
|---|---:|---|---|
| `ttc_apc_clean_data.csv` | **425,591** | one (trip, stop) | demand, OD, dwell, headway, travel time |
| `ttc_apc_data.csv` | **573,596** | one (trip, stop) | **geometry** (has Lat/Lon) |
| `ttc_avl_seg_clean_data.csv` | **604,448** | one GPS ping (segmented) | instantaneous **speed** (KPH), travel-time / bunching cross-check |
| `ttc_avl_data.csv` | **6,200,000** | one raw GPS ping | raw source; usually **not** needed |
| `20240401_Vehicle GPS Rt29 RT 929 Nov 2023.csv` | — | one raw GPS ping | vendor export; superseded by the two AVL files above |
| GTFS: `stops.txt` | 9,464 | one stop | stop names / lat-lon (2020 — see GAP 4) |
| GTFS: `trips.txt` | — | one scheduled trip | shape_id ↔ trip mapping |
| GTFS: `stop_times.txt` | — | one (trip, stop) | scheduled times, `shape_dist_traveled` |
| GTFS: `shapes.txt` | — | one shape polyline point | route polyline for plotting |
| GTFS: `routes.txt` | — | one route | route_id lookup (29 = **61327**) |
| GTFS: `calendar.txt` / `calendar_dates.txt` | — | one service pattern | **2020 dates** — provenance flag |
| GTFS: `agency.txt` | 1 | the agency | TTC metadata |

Load any of these with `common.py`:

```python
import common as C
apc   = C.load_apc_clean()          # ttc_apc_clean_data.csv
apc_x = C.load_apc_raw()            # ttc_apc_data.csv  (Lat/Lon)
avl   = C.load_avl_seg()            # ttc_avl_seg_clean_data.csv
stops = C.load_gtfs("stops.txt")   # any GTFS table by name
```

---

## 2 · `ttc_apc_clean_data.csv` — the workhorse (demand / OD / dwell / headway)

**425,591 rows. One row per (trip, stop).** This is the cleaned Automatic
Passenger Count table — the source for almost everything *except* geometry.

### Key columns

| Column | Type | Units | Meaning |
|---|---|---|---|
| `DAYTYPE` | str (cat) | — | Service day class: `MoTuWeThFr`, `Sa`, `Su`. |
| `Route` | int | — | Always `29`. |
| `Branch` | str (cat) | — | `DLWI`, `29Dcon`, `DLPRcon` (see §7 — branches differ!). |
| `VehicleID` | int | — | Physical bus. |
| `RouteDirection` | str (cat) | — | `NORTH` or `SOUTH`. |
| `StopID` | int | — | Stop identifier. **Cast to `str`** to match `node_ids`/`DataLoader`. |
| `ONSTREET` | str | — | Street the stop is on (human label). |
| `ATSTREET` | str | — | Cross street (human label). |
| `PeriodID` | str (cat) | — | Time-of-day bucket (see value set below). |
| `StopArrivalTime` | str | datetime | **Full timestamp** `YYYY-MM-DD HH:MM:SS`, e.g. `2023-11-01 04:48:07`. Parse with `pd.to_datetime`. |
| `StopDepartureTime` | str | datetime | Same format; departure from the stop. **Headway source** at `StopSeq == 1`. |
| `NoTrips` | int | count | Number of trips this aggregated row represents (often 1). |
| `AvgDwell` | float | **seconds** | Dwell time at the stop. **Dwell source.** |
| `AvgOTP` | float | — | On-time-performance metric. |
| `Arrload` | float | pax | Passenger load **on arrival**. |
| `Boarding` | float | pax | Boardings at this stop. **Demand / OD row-margin source.** |
| `Alighting` | float | pax | Alightings at this stop. **OD column-margin source.** |
| `Depload` | float | pax | Passenger load **on departure** (`Arrload + Boarding − Alighting`). |
| `TripID` | str/int | — | Groups the stops of one vehicle run. **Travel-time grouping key.** |
| `StopSeq` | int | — | Stop order within the trip, `1..N`. `StopSeq == 1` is the terminal. |
| `AbnormalAvgDwell` | flag | — | QA flag: dwell looks abnormal. |
| `LoadRate` | float | — | Load ÷ capacity. |
| `AbnormalLoadRate` | flag | — | QA flag: load looks abnormal. |
| `SteadyState` | flag | — | Trip is in steady-state (not start/end warm-up). |
| `BunchState` | flag | — | Bus-bunching indicator. |

### NORTH shape sanity-check
NORTH has **49 unique `StopID`s** with `StopSeq` running **1..49**, so the env you
regenerate from APC has **49 nodes**. The shipped *reference* env in
`setup/ttc_route_29_data/` has only **48** (46 stops + 2 terminals) because it
dropped/merged one low-traffic stop. Neither `C.stop_sequence()` nor Day 2 reduces
49→48 — keeping all 49 unique stops is the more faithful result, and the +1 vs the
reference is expected. (SOUTH matches the reference at 50.)

### How to use it
- **Demand λ:** `Boarding` grouped by `StopID`, meaned per trip, divided by headway.
- **OD matrix:** row margins from `Boarding`, column margins from `Alighting`, IPF seeded upper-triangular (Day 5).
- **Dwell:** `AvgDwell` (seconds) per stop.
- **Headway:** consecutive `StopDepartureTime` where `StopSeq == 1` (Day 4).
- **Travel time:** consecutive `StopArrivalTime` within one `TripID` (Day 3).

---

## 3 · `ttc_apc_data.csv` — same table **plus Lat/Lon** (geometry source)

**573,596 rows. One row per (trip, stop).** This is the *raw* APC export. It has
the same passenger columns as the clean file, but adds the two columns you need
for **real geometry**.

### The columns that matter here

| Column | Type | Units | Meaning |
|---|---|---|---|
| *(unnamed first col)* | int | — | Pandas will read it as `Unnamed: 0` — a leftover index. Ignore it. |
| `Lat` | float | degrees | Stop latitude. **~100% coverage on Route 29.** |
| `Lon` | float | degrees | Stop longitude. **~100% coverage on Route 29.** |
| `RouteName` | str | — | Human route name (extra vs. the clean file). |
| `AvgSample` | float | — | Sample size behind the averages (extra vs. clean file). |

Everything else (`DAYTYPE`, `Route`, `Branch`, `VehicleID`, `RouteDirection`,
`StopID`, `ONSTREET`, `ATSTREET`, `PeriodID`, `StopArrivalTime`,
`StopDepartureTime`, `NoTrips`, `AvgDwell`, `AvgOTP`, `Arrload`, `Boarding`,
`Alighting`, `Depload`) means the same as in §2.

### How to use it — real spacing (fixes GAP 2)
The placeholder in `setup/ttc_route_29_data/dataloader.py` estimates spacing as
`tt_mean × (20 km/h)`. **Do not do that.** Instead:

```python
ll  = C.stop_latlon(C.load_apc_raw(), "NORTH")   # StopID, Lat, Lon (median per stop)
seq = C.stop_sequence(C.load_apc_clean(), "NORTH")  # canonical order
# merge on StopID, then cumulative C.haversine(...) along the sequence
```

True inter-stop spacing = **cumulative haversine distance along the canonical stop
sequence**. Key each link by its **downstream** stop id (the `DataLoader`
convention). See Day 2 (`day2_geometry.py`).

> **Note:** GTFS `shapes.txt` also carries `shape_dist_traveled`, which follows
> the road (not straight lines). It is a nice cross-check, but it comes from the
> 2020 feed (GAP 4) — prefer the 2023 APC Lat/Lon as ground truth.

---

## 4 · `ttc_avl_seg_clean_data.csv` — GPS pings (speed + travel-time cross-check)

**604,448 rows. One row per segmented GPS ping.** Cleaned, segmented AVL. Use it
for **instantaneous speed** and as a **cross-check** on travel time and bunching.

### Key columns

| Column | Type | Units | Meaning |
|---|---|---|---|
| `Date_Key` | int | YYYYMMDD | Service date. |
| `Message_DateTime` | str | datetime | Timestamp of the ping. |
| `Vehicle` | int | — | Bus. |
| `Latitude`, `Longitude` | float | degrees | Ping position. |
| `Heading` | float | degrees | Compass bearing. |
| `Route` | int | — | 29. |
| `Run` | int | — | Vehicle run/block number. |
| `Destination` | str | — | Headsign at the time. |
| `OffRoute` | flag | — | Ping flagged off the route geometry. |
| `Delayed` | flag | — | Delay indicator. |
| `KPH` | float | **km/h** | Instantaneous speed. Mean ≈ **13.3**, std ≈ 15, max **97**. **Speed source.** |
| `tatripId` | int | — | Transit-agency trip id. |
| `TripID` | str | — | Composite id, e.g. `1000_N_0_29Dcon` (run_direction_seq_branch). |
| `Distance` | float | meters | ⚠️ **CORRUPT — do not trust** (see caveat below). |
| `SteadyState` | flag | — | Steady-state indicator. |
| `RouteDirection` | str (cat) | — | `NORTH` / `SOUTH`. |
| `AROUND_TIME`, `AroundLat`, `AroundLon` | — | — | Nearest-stop matching helpers. |
| `Time_diff_abs` | float | seconds | Time gap used in segmentation. |
| `BunchedTrips` | flag/int | — | Bunching indicator between trips. |

### ⚠️ The `Distance` column is corrupt
`Distance.max()` is **9,163,817 m** (≈9,164 km on a route a few km long). Some
segmentation resets accumulate garbage. **Ignore `Distance` for spacing** — use
APC Lat/Lon (§3) or recompute distance yourself from `Latitude`/`Longitude` with
`C.haversine`.

### ⚠️ `KPH == 0` means *dwelling*, not missing
Many pings read `KPH = 0`. That is a bus **stopped at a stop or a light**, not a
bad record. If you want a "moving speed", filter to `KPH > 0` (or a small
threshold); if you want realistic door-to-door time, keep the zeros because
dwell is part of the journey. Decide consciously — it's a modelling choice.

---

## 5 · `ttc_avl_data.csv` — raw unsegmented GPS (usually skip)

**≈6.2M rows. One row per raw GPS ping.** Same idea as §4 but *unsegmented* and
much larger, with a leading unnamed index column and no `TripID`/`RouteDirection`
segmentation columns. You rarely need this directly — the segmented file (§4)
already did the hard work. Reach for it only if you must re-derive segmentation
from scratch (out of scope for the core deliverable).

> `20240401_Vehicle GPS Rt29 RT 929 Nov 2023.csv` is the vendor's original
> export (note the UTF-8 BOM on its header). It is the ancestor of
> `ttc_avl_data.csv`; prefer the cleaned/segmented file.

---

## 6 · GTFS static feed — schedule, names, polylines (cross-check only)

GTFS is the standard transit schedule format: a set of `.txt` CSV tables.

| File | Key columns | Use |
|---|---|---|
| `stops.txt` | `stop_id`, `stop_code`, `stop_name`, `stop_lat`, `stop_lon`, `location_type`, `parent_station` | Stop names & lat-lon **cross-check**. 9,464 stops (all TTC, not just 29). |
| `routes.txt` | `route_id`, `route_short_name`, `route_long_name`, `route_type` | Find route 29: `route_short_name == "29"`, `route_long_name == "DUFFERIN"`, **`route_id == 61327`** (= `C.GTFS_ROUTE_29_ID`). |
| `trips.txt` | `route_id`, `service_id`, `trip_id`, `trip_headsign`, `direction_id`, `shape_id` | Map a route/direction to its `shape_id`s. |
| `stop_times.txt` | `trip_id`, `arrival_time`, `departure_time`, `stop_id`, `stop_sequence`, `shape_dist_traveled` | Scheduled times & cumulative on-road distance. Note `shape_dist_traveled` here is in the feed's unit (small floats). |
| `shapes.txt` | `shape_id`, `shape_pt_lat`, `shape_pt_lon`, `shape_pt_sequence`, `shape_dist_traveled` | Route **polyline** for plotting the line on a map. |
| `calendar.txt` | `service_id`, `monday..sunday`, `start_date`, `end_date` | Which days a `service_id` runs — **but see the date flag below.** |
| `calendar_dates.txt` | `service_id`, `date`, `exception_type` | Service exceptions (holidays). |
| `agency.txt` | agency metadata | TTC info. |

### GTFS time format
`arrival_time` / `departure_time` are `HH:MM:SS` **strings that can exceed 24h**
(e.g. `25:10:00` for after-midnight service). Don't parse them as wall-clock
without handling the >24h case.

---

## 7 · Data-quality caveats (memorize these)

These are the traps that separate a correct pipeline from a plausible-looking
wrong one.

### 7.1 GAP 4 — GTFS is **2020**, the AVL/APC is **2023**
`calendar.txt` spans **`20201011 .. 20201121`** — an **October–November 2020**
snapshot. The APC/AVL data is **November 2023**. Consequences:
- Only **~72%** of the 2023 APC Route-29 `StopID`s appear in the 2020 GTFS
  `stops.txt`. Stops were added/moved/removed in the 3-year gap.
- **Therefore:** treat GTFS as a **cross-check and a source of shape polylines**,
  *not* as ground truth for the 2023 stop set or geometry. When APC and GTFS
  disagree, **APC 2023 wins.**
- `StopID` (APC) and `stop_id` (GTFS) are the same identifier space where they
  overlap, but do not assume every APC stop has a GTFS match.

### 7.2 Corrupt AVL `Distance`
`ttc_avl_seg_clean_data.csv.Distance` reaches **~9.16 million meters**. **Never
use it for spacing.** Recompute from lat/lon or use APC geometry.

### 7.3 `KPH == 0` = dwelling
Zeros in `KPH` are stopped buses, not nulls. Filter or keep them on purpose (§4).

### 7.4 Multiple branches don't share all stops
Route 29 runs three branches: **`DLWI`**, **`29Dcon`**, **`DLPRcon`**. They do
**not** all serve the same stops, so a naive stop list mixes branch-specific
stops. `C.stop_sequence()` canonicalizes by **median `StopSeq`**, which is a
reasonable default — but deciding whether to model a single "trunk" or keep
branch stops is a **real modelling choice** (see Day 2 in
[`04_week_plan.md`](04_week_plan.md), and the `# STUDENT TODO` hooks in the
scripts).

### 7.5 `StopID` dtype
`StopID` reads as an integer but the env / `DataLoader` uses **string** node ids.
`common.py` already casts to `str` in its helpers — do the same everywhere you
build keys, or your merges will silently miss.

### 7.6 Abnormal / warm-up rows
`AbnormalAvgDwell`, `AbnormalLoadRate`, and `SteadyState` flag rows that are
outliers or trip warm-up. Consider filtering on `SteadyState` and dropping
abnormal-flagged rows before fitting distributions.

---

## 8 · Categorical value sets (verified from the data)

| Column | Values |
|---|---|
| `Route` | `29` (only) |
| `RouteDirection` | `NORTH`, `SOUTH` |
| `Branch` | `DLWI`, `29Dcon`, `DLPRcon` |
| `DAYTYPE` | `MoTuWeThFr`, `Sa`, `Su` |
| `PeriodID` | `Early morning`, `AM peak`, `Morning`, `Midday`, `Afternoon`, `PM peak`, `Early evening`, `Late evening` |

> `PeriodID` and `DAYTYPE` are your handles for **stratifying** derivations
> (e.g. peak vs off-peak headway/demand). The default deliverable is
> **month-wide aggregates**; period/day stratification is a documented extension.

---

## 9 · "Which file for which job?" — quick reference

| I need… | Use this file | Key columns | Script |
|---|---|---|---|
| **Geometry / spacing** | `ttc_apc_data.csv` | `Lat`, `Lon` + canonical stop order | `day2_geometry.py` |
| **Travel time (stop→stop)** | `ttc_apc_clean_data.csv` (primary) | `TripID`, `StopSeq`, `StopArrivalTime` | `day3_travel_time.py` |
| **Travel time / speed cross-check** | `ttc_avl_seg_clean_data.csv` | `KPH`, `Latitude`, `Longitude` | (cross-check) |
| **Instantaneous speed** | `ttc_avl_seg_clean_data.csv` | `KPH` (drop the corrupt `Distance`) | (cross-check) |
| **Dispatching headway** | `ttc_apc_clean_data.csv` | `StopDepartureTime` where `StopSeq == 1` | `day4_headway.py` |
| **Per-stop demand λ** | `ttc_apc_clean_data.csv` | `Boarding` ÷ headway | `day5_demand_od.py` |
| **OD matrix** | `ttc_apc_clean_data.csv` | `Boarding` (row) + `Alighting` (col) → IPF | `day5_demand_od.py` |
| **Dwell time** | `ttc_apc_clean_data.csv` | `AvgDwell` (seconds) | `day5_demand_od.py` / build |
| **Stop names / labels** | `ttc_apc_clean_data.csv` or GTFS `stops.txt` | `ONSTREET`, `ATSTREET` / `stop_name` | any |
| **Route polyline for a map** | GTFS `shapes.txt` (+`trips.txt`) | `shape_pt_lat/lon` | (plotting) |
| **Bunching / segmentation** | `ttc_avl_seg_clean_data.csv` | `BunchState`, `BunchedTrips`, `SteadyState` | (analysis) |

---

## 10 · Counts → rates cheat-sheet

Everything raw is a **count over the November window**. Convert deliberately:

| Raw quantity | Divide by | Gives |
|---|---|---|
| Mean `Boarding` per trip at stop *i* | `dispatching_headway_sec` | λ_i (pax/sec boarding rate) |
| Total OD count *i→j* (post-IPF) | `num_trips × headway_sec` | OD rate (pax/sec) |
| `AvgDwell` | — (already seconds) | dwell seconds |
| Stop→stop Δ`StopArrivalTime` | — (already seconds) | travel time seconds |
| `KPH` | ÷ 3.6 | m/s |

Because λ and OD rates both divide by the **headway**, fixing the headway
placeholder (GAP 1) rescales the whole demand model (**GAP 5**). Keep that
coupling explicit — it is spelled out in
[`03_data_to_env_mapping.md`](03_data_to_env_mapping.md).

---

### Done looks like
You can, from memory, answer: *which file has Lat/Lon?* (`ttc_apc_data.csv`),
*which column is corrupt?* (`Distance` in the segmented AVL), *why is GTFS only a
cross-check?* (2020 vs 2023, ~72% overlap), and *what do I divide boardings by to
get λ?* (the dispatching headway in seconds). If so, move on to
[`03_data_to_env_mapping.md`](03_data_to_env_mapping.md).
