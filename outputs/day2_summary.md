1. Inspect the spacing table it writes. Confirm the terminal (seq 1) has no spacing entry (spacing is keyed by the downstream stop, matching the DataLoader convention).

The start terminal has a NaN spacing_real_m:
 stop_seq StopID        ONSTREET                                   ATSTREET  spacing_real_m  cum_dist_m
      1.0  11991    Strachan Ave                        Fleet St North Side             NaN         0.0

2. Compare real spacing vs the fake 20 km/h spacing the script prints side by side. Note where they diverge most (long express-ish links vs short dense-stop links).

There is large divergence in the first link (stop_seq 2.0) and in the 4th link (stop_seq 5.0). Both are an example of the
express link effect (though the former could also be explained by longer dwell times before bus commences trip). The large relative error of link 5 (stop_seq 6.0) can be explained by the dense-stop link effect.

 stop_seq StopID  spacing_real_m  placeholder_m  abs_err_m
      2.0  14537           446.9         1032.3      585.4
      3.0  14538           206.0          178.7       27.3
      4.0  14539           274.3          327.0       52.7
      5.0  16098           524.9          728.9      204.0
      6.0   2066            91.3          148.8       57.4
      7.0   9739           161.9          226.9       65.0
      8.0   2081           210.2          242.9       32.7
      9.0   2014           288.5          344.5       56.0
     10.0   2057           252.5          186.3       66.2
     11.0   2043           318.2          345.4       27.2
     12.0   2033           394.4          391.7        2.6
     13.0   2098           257.7          240.1       17.6


3. Look at the generated route map (stops plotted at their Lat/Lon, connected in sequence). Does it look like a real north–south corridor, or is a stop obviously mis-ordered? (If a stop jumps around, that's a branch artifact — see the TODO hook.)

The generated route maps look like real north-south corridors.
