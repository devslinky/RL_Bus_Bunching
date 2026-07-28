1. Inspect the segment table. Which links are slowest / most variable (highest CV)? High CV links are exactly where bus bunching is born — note them.

For direction North: Link 5 (from stop ids 16098 to 2066) and Link 7 (from stop ids 9739 to 2081) are the most variable (highest CV).

2. Cross-check with AVL speed: convert your tt_mean and the Day-2 spacing into an implied speed (spacing_m / tt_mean × 3.6 km/h) and compare against the AVL KPH median for the corridor (~13 km/h). They should be in the same ballpark; a wild mismatch flags a bad segment.

For direction north; link 1 (stop id 11991 to 14537) has tt_mean of ~185 sec and a distance of ~447 m; therefofe the implied speed for this link is 446/185 * 3.6 = 8.68 km/hr which is near the 13 km/hr mean AVL speed.

3. Write a short note on CV: why link travel-time CV matters for a bunching simulator (it's the noise that destabilizes headways), and whether CV varies by PeriodID.

The greater the travel-time variance; the more likely that the transit system itself will experience non-determenistic and increasingly variable travel times between links; allowing for bus bunching to occur (leading buses may experience delays while following buses may experience express times allowing bunching to occur). Travel Time is shown to vary by period; and the variance of each period is also different amongst them so CV also varies by PeriodID.

======================================================================
DAY 3 — TRAVEL TIME  (NORTH)
======================================================================
    from_stop_id to_stop_id       PeriodID     tt_mean     tt_std     tt_cv  tt_count
0          11991      14537        AM peak  149.578853  84.709261  0.566318       558
1          11991      14537      Afternoon  245.746552  89.460885  0.364037       580
2          11991      14537  Early evening  194.232831  92.508812  0.476278       597
3          11991      14537  Early morning   94.040000  32.614167  0.346812       125
4          11991      14537   Late evening  118.005025  54.421716  0.461181       597
..           ...        ...            ...         ...        ...       ...       ...
379         2104       2108  Early morning   18.580247   7.653788  0.411931       162
380         2104       2108   Late evening   18.165517  20.965722  1.154149       580
381         2104       2108         Midday   25.517204  10.829246  0.424390       930
382         2104       2108        Morning   25.376190  10.077993  0.397144       210
383         2104       2108        PM peak   26.268382   9.373056  0.356819       544

======================================================================
DAY 3 — TRAVEL TIME  (SOUTH)
======================================================================
    from_stop_id to_stop_id       PeriodID     tt_mean      tt_std     tt_cv  tt_count
0           2107       2105        AM peak   33.044597   15.259656  0.461790       583
1           2107       2105      Afternoon   38.263328   17.690803  0.462344       619
2           2107       2105  Early evening   32.125887   16.097177  0.501066       564
3           2107       2105  Early morning   30.314286   35.464662  1.169899       140
4           2107       2105   Late evening   26.857143   13.050467  0.485922       553
..           ...        ...            ...         ...         ...       ...       ...
387        16384      11992  Early morning   87.288889   21.280945  0.243799       135
388        16384      11992   Late evening  102.646503   59.582177  0.580460       529
389        16384      11992         Midday  160.659259   70.595112  0.439409       945
390        16384      11992        Morning  236.831169  158.171047  0.667864       231
391        16384      11992        PM peak  218.584532  106.174487  0.485737       556