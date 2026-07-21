# 01 · Architecture: how the simulator consumes your data

> **Read this first.** Before you touch a single row of raw data, you need to know
> *what shape* the simulator expects your numbers to arrive in. This doc is a guided
> tour of the setup/ building blocks and the one narrow interface — the **DataLoader** —
> that your ETL pipeline has to satisfy. Everything you derive in
> [`04_week_plan.md`](04_week_plan.md) exists to fill the fields described here.

**Where you are in the docs:**
`README.md` → **01 architecture (you are here)** → [`02_data_dictionary.md`](02_data_dictionary.md) (the raw columns) → [`03_data_to_env_mapping.md`](03_data_to_env_mapping.md) (which raw column becomes which field, and the 5 gaps) → [`04_week_plan.md`](04_week_plan.md) (day-by-day) → [`05_deliverables_checklist.md`](05_deliverables_checklist.md).

All paths below are relative to the repo root `/home/jiahao/Documents/busoperation`. Line numbers are exact at time of writing — if they drift by a line or two after edits, search for the quoted symbol.

---

## 1. The 30-second mental model

```
raw_data/*.csv,*.txt   →  YOUR ETL  →  setup/ttc_route_29_data/  →  DataLoader  →  Network + Route_Schema  →  Blueprint  →  Simulator
(APC / AVL / GTFS)        (day2..5)     data_<dir>.pickle           (thin reader)   (setup/ttc_route_29.py)     (bundles both)   (runs the buses)
                          build_env_data  summary_<dir>.json
```

Read the arrow chain right-to-left to understand *dependencies*, left-to-right to
understand *your work*. The simulator only ever talks to a `Blueprint`. A `Blueprint`
is built from a `Network` + a `Route_Schema`. Those two classes read **every number**
from a `DataLoader`. The `DataLoader` reads two files that **your pipeline regenerates**.
So the entire contract you must satisfy is: *"produce a `data_<dir>.pickle` whose keys
match what the DataLoader properties return."* That's it. Everything downstream is
already wired.

---

## 2. Blueprint — the single object the simulator consumes

**File:** `setup/blueprint.py`

`Blueprint(env_name)` (`setup/blueprint.py:25`) is the one object the whole simulator
depends on. Its constructor is a dispatch table on `env_name`:

```python
# setup/blueprint.py:36-41
elif self.env_name == 'ttc_route_29_north':
    self.network = TTC_Route_29_North_Network()
    self.route_schema = TTC_Route_29_North_Route_Schema()
elif self.env_name == 'ttc_route_29_south':
    self.network = TTC_Route_29_South_Network()
    self.route_schema = TTC_Route_29_South_Route_Schema()
```

Recognised `env_name`s include `"cd_route_3"` (the Chengdu reference), and
`"ttc_route_29_north"` / `"ttc_route_29_south"`. After picking a `Network` and a
`Route_Schema`, the Blueprint precomputes three convenience maps from them:

| Blueprint attribute | Built by | What it is |
|---|---|---|
| `_route_node_to_link`, `_route_link_to_node` | `_generate_node_and_link_map` (`blueprint.py:123`) | "next link for this node" / "next node for this link" lookups |
| `_route_node_distance` | `_generate_node_distance_from_terminal` (`blueprint.py:152`) | cumulative distance of each node from the start terminal |
| `_route_stop_arrival_rate` | `_calculate_total_arrival_rate` (`blueprint.py:193`) | per-stop total arrival rate = **sum of each OD-table row** |

Two facts worth internalising:

- `_calculate_total_arrival_rate` (`blueprint.py:193-211`) **sums the OD table by
  origin row** to get each stop's total boarding rate, and then forces the *last*
  stop's rate to `0.0` (nobody boards at the final stop). This is why your OD matrix
  and your per-stop lambda must be **mutually consistent** — the row sums of the OD
  rate table *are* the effective lambdas the sim uses.
- `_get_distance` (`blueprint.py:176`) currently uses Manhattan distance on node
  `(x, y)` coordinates and carries a `# TODO ... should be replaced by real travel
  distance`. Since the TTC network lays stops out on a straight line (`y = 0`, x =
  cumulative spacing), this reduces to your spacing. **Better spacing → better
  distances** — that is exactly GAP 2 (see [`03_data_to_env_mapping.md`](03_data_to_env_mapping.md)).

**What today's Blueprint builds** (verified, `Blueprint("ttc_route_29_north")`):
48 nodes (46 stops + 2 terminals), 47 links, an OD table with 49 origins,
`terminal_start 11991 → terminal_end 2108`, `schedule_headway (300, 60)`. That
`(300, 60)` is the placeholder you are going to replace (GAP 1).

---

## 3. Network — the geometry + travel-time graph

**File:** `setup/network.py` (abstract) and `setup/ttc_route_29.py` (concrete)

`Network` (`setup/network.py:11`) is an abstract class wrapping a
`networkx.DiGraph` (`self._G`). Subclasses implement exactly one method,
`_define_network` (`network.py:180`), whose job is to add every node and edge to
`_G`. The base class then exposes the graph through four read-only properties the
Blueprint/Simulator rely on:

- `terminal_node_geometry_info` (`network.py:72`) — `{node_id → TerminalNodeGeometry}`
- `stop_node_geometry_info` (`network.py:86`) — `{node_id → StopNodeGeometry}`
- `link_geometry_info` (`network.py:100`) — `{link_id → LinkGeometry}`
- `link_distribution` (`network.py:115`) — `{link_id → LinkDistribution}`

### How `_define_network` is filled for TTC

Look at `TTC_Route_29_Network._define_network` (`setup/ttc_route_29.py:39-113`). It:

1. Pulls `node_ids`, `link_time_info`, and `get_spacing()` from the DataLoader
   (`ttc_route_29.py:41-43`).
2. Walks the ordered `node_ids`, accumulating an `x` coordinate by adding each link's
   **spacing** (`ttc_route_29.py:52-58`); every node is placed at `y = 0`.
3. Adds each node as a `terminal` (first & last id) or `stop` node with a
   `StopNodeGeometry(x, y, berth_num=2)` (`ttc_route_29.py:66-76`).
4. For each consecutive pair, builds a `LinkGeometry(head, tail, x_head, 0, spacing)`
   and a `LinkDistribution(tt_mean, tt_cv, 'normal')`, reading
   `tt_mean = link_time_info[tail]['loc']` and
   `tt_std = link_time_info[tail]['scale']`, with `tt_cv = tt_std/tt_mean`
   (`ttc_route_29.py:82-113`).

Two placeholders live right here and are yours to fix:

- **Spacing** comes from `DataLoader.get_spacing()`, which today fakes distance as
  `tt_mean × (20 km/h)` — **GAP 2**. Real haversine spacing from APC Lat/Lon replaces it.
- If a stop has no travel-time entry, the code falls back to `tt_mean = 50, tt_cv =
  0.2` (`ttc_route_29.py:96-98`). Your Day-3 output should leave as few of those
  defaults as possible.

### Node/edge attribute keys (don't rename these)

`_define_network` must attach exactly these keys, because the base-class properties
read them by name:

- nodes: `node_type='terminal'|'stop'`, plus `terminal_node_geometry=` **or**
  `stop_node_geometry=`.
- edges: `link_id=`, `link_geometry=`, `link_distribution=`.

---

## 4. Route_Schema — demand, headway, terminals, boarding

**File:** `setup/route.py` (abstract) and `setup/ttc_route_29.py` (concrete)

`Route_Schema` (`setup/route.py:25`) is the second half of a Blueprint. Its
constructor (`route.py:49-102`) calls a series of `_define_*` abstract methods, one
per route attribute, and packs the results into a `Route_Details` dataclass
(`route.py:8-22`) per route id:

```python
# setup/route.py:8-22  — the Route_Details dataclass
route_id: str
terminal_id: str
visit_seq_stops: List[str]
end_terminal_id: str
od_rate_table: Dict[str, Dict[str, float]]   # origin -> dest -> pax/sec
schedule_headway: float                       # seconds
schedule_headway_std: float                   # seconds
boarding_rate: Dict[str, float]               # stop -> pax/sec service rate
bus_capacity: int
hold_stops: List[str]
```

### The `_define_*` methods you must know

| Abstract method (`route.py`) | Fills `Route_Details` field | TTC implementation (`ttc_route_29.py`) |
|---|---|---|
| `_define_route_ids` (`:145`) | `route_id` | `['29N']` or `['29S']` (`:172-175`) |
| `_define_od_table` (`:149`) | `od_rate_table` | inferred OD from DataLoader, else uniform fallback (`:139-170`) |
| `_define_schedule_headway` (`:153`) | `schedule_headway`, `_std` | `(H_mean, H_std)` from DataLoader (`:177-180`) |
| `_define_terminal` (`:157`) | `terminal_id` | `node_ids[0]` (`:182-185`) |
| `_define_visit_seq_stops` (`:161`) | `visit_seq_stops` | `node_ids[1:-1]` (`:187-190`) |
| `_define_end_terminal` (`:165`) | `end_terminal_id` | `node_ids[-1]` (`:192-195`) |
| `_define_boarding_rate` (`:169`) | `boarding_rate` | `1/4.0` pax/sec at every stop (`:197-206`) |
| `_define_hold_stops` (`:173`) | `hold_stops` | every stop is holdable (`:208-212`) |
| `_define_bus_capacity` (`:177`, optional) | `bus_capacity` | not overridden → `INF` |

Key subtlety in `_define_od_table` (`ttc_route_29.py:139-170`): it **prefers the
inferred OD matrix** from `DataLoader.od_rate_table`; only if that dict is empty does
it fall back to spreading each stop's `stop_pax_arrival_rate` uniformly over
downstream stops. Your Day-5 IPF matrix is what lands in that `od_rate_table` — so
the quality of your OD inference directly shapes the simulated demand.

`Route_Schema` also exposes `route_details_by_id`, `route_OD_rate_table`, `terminal`,
`end_terminal`, and `terminal_to_routes_info` as properties (`route.py:104-135`) — the
Simulator's builder reads those, but you never write them directly; they are derived
from the `_define_*` outputs above.

---

## 5. The geometry / distribution dataclasses

**File:** `setup/config_dataclass.py`

These are the tiny frozen records the Network fills. You will construct the first four
constantly; know their fields:

```python
# setup/config_dataclass.py
TerminalNodeGeometry(x, y)                       # :5   a terminal's position
StopNodeGeometry(x, y, berth_num)                # :16  a stop's position + #berths
LinkGeometry(head_node, tail_node, x_head, y_head, length)  # :41  a link's endpoints + length (meters)
LinkDistribution(tt_mean, tt_cv, tt_type)        # :50  travel-time mean, coeff-of-variation, dist name
```

Two gotchas:

- **`LinkGeometry.length` is your spacing in meters.** `LinkDistribution.tt_mean` is
  travel time in seconds. They are independent inputs — do not confuse distance and time.
- **`LinkDistribution` takes `tt_cv`, not `tt_std`.** Your Day-3 script derives both
  `tt_mean` and `tt_std`; the DataLoader stores `{'loc': mean, 'scale': std}` and the
  Network computes `tt_cv = scale / loc` (`ttc_route_29.py:94`). Keep the units in
  seconds throughout.

(`StopNodeOperation`, `PaxOperation`, `TerminalNodeOperation` also live in this file
but are set elsewhere in the simulator — you don't populate them from data.)

---

## 6. The TTC classes and the DataLoader that injects data

**Files:** `setup/ttc_route_29.py` and `setup/ttc_route_29_data/dataloader.py`

`setup/ttc_route_29.py` defines `TTC_Route_29_Network` + `TTC_Route_29_Route_Schema`
and thin per-direction subclasses (`..._North_Network`, `..._South_Route_Schema`,
etc., `ttc_route_29.py:215-241`). **Neither class computes any numbers itself** — each
constructs a `DataLoader(direction)` (`ttc_route_29.py:36` and `:125`) and reads every
value from it. This is the seam you work behind: *change the DataLoader's inputs and
the environment changes, without editing `ttc_route_29.py` at all.*

`DataLoader(direction)` (`setup/ttc_route_29_data/dataloader.py:14`) is deliberately
dumb: in `__init__` it just `pickle.load`s `data_<dir>.pickle` and reads
`summary_<dir>.json` (`dataloader.py:31-36`), then exposes the pieces as properties:

| DataLoader member | `dataloader.py` | Returns |
|---|---|---|
| `node_ids` | `:38` | ordered `[terminal, stop, …, stop, terminal]` (strings) |
| `stop_pax_arrival_rate` | `:44` | `{stop_id → pax/sec}` (lambda) |
| `link_time_info` | `:49` | `{tail_stop_id → {'loc': mean_s, 'scale': std_s}}` |
| `stop_info` | `:59` | list of per-stop dicts (boardings/alightings/…) |
| `link_info` | `:64` | list of per-link dicts (tt + spacing source) |
| `terminal_start_id` | `:69` | start terminal id |
| `terminal_end_id` | `:74` | end terminal id |
| `od_rate_table` | `:96` | `{origin → {dest → pax/sec}}`, upper-triangular |
| `dispatching_headway` | `:79` | `(mean_s, std_s)` — **hardcoded `(300, 60)` today** |
| `get_spacing()` (method) | `:105` | `{tail_stop_id → meters}` — **faked as `tt_mean × 20km/h` today** |
| `num_stops` | `:91` | `len(node_ids) - 2` |

The two bold rows are **GAP 1** and **GAP 2** — the placeholder logic your pipeline
eliminates. Your `build_env_data.py` writes a *new* `data_<dir>.pickle` (into
`student_project/outputs/derived/…`, never into `setup/`) with real headway and real
spacing baked in. On **Day 6** *you* install that bundle as a real-data env under `setup/`
(your design) — a DataLoader that reads it with the `(300, 60)` return and the `20 km/h`
estimate gone — and run the simulator on it. The precise mapping is in
[`03_data_to_env_mapping.md`](03_data_to_env_mapping.md); the integration guide (open-ended)
in [`06_integrate_into_simulator.md`](06_integrate_into_simulator.md).

---

## 7. Field → DataLoader-property map (the contract, at a glance)

This is the table to keep open while you build the pipeline. **Left column = a field
the environment needs; right column = the DataLoader property that supplies it.** If
your regenerated pickle produces the right-column values, the whole left column fills
itself.

| Consumed by | Network / Route_Schema field | Filled from DataLoader property | Site (`ttc_route_29.py`) |
|---|---|---|---|
| Network | node `(x,y)` layout + `LinkGeometry.length` | `get_spacing()` | `:43`, `:52-58`, `:82-104` |
| Network | ordered node set (terminals + stops) | `node_ids` | `:41`, `:63-76` |
| Network | `LinkDistribution.tt_mean` / `tt_cv` | `link_time_info` (`loc`,`scale`) | `:42`, `:90-101` |
| Route_Schema | `terminal_id` / `end_terminal_id` | `node_ids[0]` / `node_ids[-1]` (≡ `terminal_start_id`/`terminal_end_id`) | `:130-131`, `:182-195` |
| Route_Schema | `visit_seq_stops` | `node_ids[1:-1]` | `:132`, `:187-190` |
| Route_Schema | `od_rate_table` | `od_rate_table` (fallback: `stop_pax_arrival_rate`) | `:139-170` |
| Route_Schema | per-stop total arrival rate (via OD row sums) | `stop_pax_arrival_rate` | `:127`, `:157-164` |
| Route_Schema | `schedule_headway`, `schedule_headway_std` | `dispatching_headway` | `:135`, `:177-180` |
| Blueprint | node distances (Manhattan on `x`) | derived from `get_spacing()` layout | `blueprint.py:152-191` |

> Note: `terminal_start_id` / `terminal_end_id` are exposed by the DataLoader, but the
> TTC classes actually read the terminals as `node_ids[0]` / `node_ids[-1]`. So keep
> your `node_ids` **ordered terminal…terminal**, and make `terminal_start_id ==
> node_ids[0]` and `terminal_end_id == node_ids[-1]` so the two never disagree.

---

## 8. The runtime loop (so you know what "good data" buys you)

**Files:** `quick_run.py`, `simulator/simulator.py`, `config.py`, `agent/agent.py`

You will not modify the simulator, but seeing the loop makes the data requirements
concrete. `config.build_simulation_elements(config_path)` (`config.py:38`) reads the
YAML, calls `Blueprint(env_name)` (`config.py:93`), and returns
`(blueprint, agent, run_config, _)`. `quick_run.py` then runs the classic loop
(`quick_run.py:59-71`):

```python
simulator = Simulator(blueprint, agent, run_config)      # simulator/simulator.py:19
for t in range(run_config['episode_duration']):
    snapshot = simulator.step(t, stop_bus_hold_action)   # advance 1 second
    stop_bus_hold_action = agent.calculate_hold_time(snapshot)
    snapshot.record_holding_time(stop_bus_hold_action)
metrics, _ = simulator.get_metrics()                     # e.g. headway_std
```

Inside `Simulator.step(t, hold_times)` (`simulator/simulator.py:231-295`) one
simulated second does, in order: (0) dispatch a bus from the terminal if the
**dispatch headway** has elapsed (`:244-257` — this is where your GAP-1 headway lives),
(1) generate passenger arrivals from **lambda / OD** (`:260-262`), (2) move buses along
**links** using the travel-time distribution (`:265-267`), (3) run **stop** boarding /
alighting using **boarding_rate** and spacing (`:270-275`), (4) apply the agent's
**holding** action (`:278-282`), (5) accrue in-vehicle delay (`:286-292`), then return a
`Snapshot`.

The agent's contract is `calculate_hold_time(snapshot) -> {(stop_id, route_id,
bus_id) → hold_seconds}` (`agent/agent.py:39`). `get_metrics()`
(`simulator/simulator.py:305`) returns the headway-variance / wait-time metrics the
whole project is trying to reduce.

**Why this matters for your data:** every number you derive shows up as a *physical
behaviour* in this loop. A wrong headway (300 vs the true ~536 s) means buses are
dispatched twice as often as reality; wrong spacing distorts the time-space diagram;
a wrong OD matrix mis-loads the buses. Getting the DataLoader inputs right is what
makes the simulation *say something true about Route 29*.

---

## 9. The reference implementation to imitate: Chengdu

**Files:** `setup/chengdu.py` and `setup/chengdu_route_3_data/dataloader.py`

Chengdu Route 3 (`env_name == "cd_route_3"`) is the **worked example** of a
data-driven environment — study it before writing your pipeline, then mirror its
shape. It follows the *identical* pattern:

- `setup/chengdu_route_3_data/dataloader.py` is a `DataLoader` exposing `node_ids`
  (`:36`), `stop_pax_arrival_rate` (`:64`, note the `/60` to convert **pax/min → pax/sec**),
  `link_time_info` (`:72`, returns each stop's fitted `norm` `{loc, scale}`),
  `spacing` (`:79`), and `dispatching_headway` (`:86`, which actually **fits a normal**
  to observed terminal departure gaps via `norm.fit` — this is the honest version of
  what your GAP-1 fix should do, versus the TTC hardcode).
- `setup/chengdu.py` reads those at module load (`:13-18`), builds `node_x_cum` from
  spacing (`:29-36`), and defines `CD_Route3_Network._define_network` (`:43-75`) and
  `CD_Route3_Route_Schema` with the same `_define_*` methods (`:82-141`).

Notice two honest touches in Chengdu you should replicate in spirit: it converts
lambda units explicitly (`/60`, `dataloader.py:68`) and it *fits* a distribution to
real departure gaps instead of hardcoding one (`dataloader.py:86-97`). Those are
exactly the disciplines GAP 1 and GAP 5 ask you to bring to TTC. See
[`03_data_to_env_mapping.md`](03_data_to_env_mapping.md) for the gap list and
[`04_week_plan.md`](04_week_plan.md) for the day-by-day build order.

---

## 10. What you must NOT break

Your pipeline is free to compute numbers however you like, but the **DataLoader's
public surface is a hard contract**. The environment (`setup/ttc_route_29.py` and,
transitively, `Blueprint` and `Simulator`) reads these by name, so the dict your
`build_env_data.py` pickles must keep producing them — same key names, same value
shapes, same units:

- **`node_ids`** — `List[str]`, ordered terminal → stops → terminal. Everything else
  keys off this ordering.
- **`stop_pax_arrival_rate`** — `{stop_id(str) → pax/sec}`. (Watch units: **per second**, not per minute.)
- **`link_time_info`** — `{tail_stop_id(str) → {'loc': mean_sec, 'scale': std_sec}}`.
  Keyed by the **downstream** stop of each link.
- **`od_rate_table`** — `{origin(str) → {dest(str) → pax/sec}}`, strictly
  upper-triangular in stop order (origin strictly upstream of dest), non-negative.
- **`terminal_start_id`**, **`terminal_end_id`** — `str`, and must equal `node_ids[0]`
  and `node_ids[-1]` respectively.
- **`link_info`** — `List[dict]` with at least `from_stop_id`, `to_stop_id`,
  `tt_mean`, `tt_std`, `tt_cv`, `spacing_m` (the last is what `get_spacing()` consumes).
- **`stop_info`** — `List[dict]` with per-stop detail (`StopID`, `stop_seq`, `lambda`,
  `boardings`, …).

Plus the two **method/property returns** you are *upgrading* (their signatures stay,
their internals change): `dispatching_headway → (mean, std)` and `get_spacing() →
{tail_stop_id → meters}`. Keep the shapes; replace the placeholder logic.

**Rename any of these keys and the environment silently breaks** — a missing key
throws at `DataLoader` load or at `Blueprint` construction, and a wrong-unit value
produces a plausible-but-wrong simulation, which is worse. When in doubt, open the
reference `setup/ttc_route_29_data/summary_north.json` and the existing pickle to see
the exact shapes you must reproduce (details in
[`02_data_dictionary.md`](02_data_dictionary.md) and
[`03_data_to_env_mapping.md`](03_data_to_env_mapping.md)).

---

### "Done" for this doc

You can answer, without looking: *Which object does the Simulator consume? Which two
classes build it? Where do their numbers come from? Which DataLoader keys can I never
rename, and what are their units?* If yes, head to
[`02_data_dictionary.md`](02_data_dictionary.md) to meet the raw columns.
