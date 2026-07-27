How many unique stops does Route 29 NORTH have, and what are the two terminals?
Route 29 North has 49 stops, with terminals being Strachan Ave @ Fleet St North Side to Dufferin St @ Wilson Ave (id = 11991 to 2108)

What are the branches (DLWI, 29Dcon, DLPRcon) and roughly how common is each?
29Dcon seems to be the default (contains all 49 stops) with 96.17% of total northbound trips falling into 29Dcon. 
DLPRcon has 0 northbound trips (this is a southbound branch with 42 stops and accounts for 0.52% of total southbound trips).
DLWI has 9 stops and accounts for 175 or 3.83% of total northbound trips.

What date range and day types (MoTuWeThFr, Sa, Su) does the APC data cover?
The APC data covers three day types : "MoTuWeThFr", "Sa", "Su" and its runs a date range from November 1st, 2023 to December 1st, 2023

Eyeball: does boarding demand vary a lot stop-to-stop? Peak vs off-peak (PeriodID)?
Boarding demand does vary significantly stop-to-stop early on in the trip; then the variation decreases near the last third of the trip. Boarding demand varies singificantly by period; however the periods of 
highest demand occur during midday and the PM peak; not during the AM peak

One sentence each on where headway, spacing, travel time, and demand will come from.
Headway will come from the APC data's real terminal-departure timestamps — consecutive StopDepartureTime values where StopSeq == 1, grouped by service date.
Spacing will come from the APC data's real Lat/Lon coordinates — cumulative haversine distance between consecutive stops along the canonical stop sequence.
Travel time will come from the APC data's per-trip timestamps — consecutive StopArrivalTime differences within the same TripID, walked in stop order.
Demand will come from the APC data's Boarding/Alighting counts — per-stop λ derived as mean boardings-per-trip ÷ real headway, and the full OD matrix derived via IPF on the boarding/alighting margins.