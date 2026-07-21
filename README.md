# TTC Route 29 — Real-Data-Grounded Simulation Environment

The bus-holding RL simulator in this repo currently runs on an environment that was
**fit to a Chengdu route from thin static inputs** — a handful of guessed numbers stand
in for real geometry, speeds, headways, and demand. Your job is to replace those guesses
with an environment **derived from real Toronto Transit Commission (TTC) Route 29
"DUFFERIN" data**: November 2023 AVL (GPS) traces and APC (automatic passenger counts),
plus a GTFS static feed. When you are done, the simulator will hold buses on a route whose
stop spacing, travel-time distributions, dispatching headway, per-stop demand, and
origin–destination flows all come from measured data instead of placeholders — a
reproducible pipeline that any future student can re-run.
