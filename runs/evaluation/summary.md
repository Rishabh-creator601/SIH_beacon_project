# Multi-seed evaluation

180 runs. Each run = one seed; random scenes also differ in beacon path, decoys, turbulence, vibration, clouds and initial pointing error.

**Time locked** = share of the whole run with a confirmed lock on the true beacon (penalises slow / failed acquisition). **Retention** = locked share after the first lock (median over runs that acquired). **False-lock runs** = runs with > 1 s locked on a wrong object.

| Suite | Scenario | Method | Runs | Acquired % [95% CI] | Acq. time median [IQR] (s) | Time locked mean ± 95% CI (%) | Runs ≥ 80% locked | Retention median (%) | False-lock runs (%) | Error median (px) | Proc. (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | random | hybrid | 40 | 92 [80-97] | 5.2 [2.2-16.8] | 71.5 ± 9.5 | 55 % | 100.0 | 8 | 3.56 | 8.1 |
| random | random | hybrid-v2 | 40 | 92 [80-97] | 5.2 [2.1-15.3] | 72.0 ± 9.6 | 57 % | 100.0 | 5 | 3.36 | 7.0 |
| random | random | none | 40 | 62 [47-76] | 2.3 [0.5-4.3] | 53.7 ± 14.4 | 50 % | 100.0 | 42 | 4.28 | 3.7 |
| scenarios | 01_clear_sky | hybrid | 10 | 100 [72-100] | 7.4 [7.4-7.4] | 87.7 ± 0.0 | 100 % | 100.0 | 0 | 1.44 | 4.9 |
| scenarios | 02_strong_turbulence | hybrid | 10 | 100 [72-100] | 11.4 [8.1-22.0] | 52.4 ± 17.9 | 20 % | 97.1 | 30 | 4.59 | 4.9 |
| scenarios | 03_occlusion | hybrid | 10 | 100 [72-100] | 27.5 [8.4-45.9] | 31.6 ± 14.3 | 0 % | 54.5 | 10 | 3.86 | 5.8 |
| scenarios | 04_decoys | hybrid | 10 | 100 [72-100] | 15.1 [14.3-15.8] | 63.7 ± 17.7 | 20 % | 100.0 | 0 | 2.70 | 5.3 |
| scenarios | 05_fast_target | hybrid | 10 | 100 [72-100] | 20.3 [9.2-30.2] | 63.1 ± 10.4 | 20 % | 99.3 | 0 | 9.63 | 7.1 |
| scenarios | 06_worst_case | hybrid | 10 | 90 [60-98] | 26.8 [15.3-43.0] | 9.7 ± 6.2 | 0 % | 29.5 | 10 | 8.23 | 21.6 |

## Random scenes, method `hybrid`: time locked (%) by scene property

| Property | Bin | Runs | Time locked mean (%) | Acquired % |
|---|---|---|---|---|
| Turbulence | 0-1 | 10 | 73.5 | 90 |
| Turbulence | 1-1.5 | 16 | 78.0 | 88 |
| Turbulence | ≥ 1.5 | 14 | 62.6 | 100 |
| Decoys | 0-0.5 | 12 | 66.7 | 83 |
| Decoys | 0.5-2.5 | 9 | 82.7 | 100 |
| Decoys | ≥ 2.5 | 19 | 69.2 | 95 |
| Beacon speed (°/s) | 0-2 | 11 | 81.7 | 100 |
| Beacon speed (°/s) | 2-3.5 | 11 | 80.1 | 100 |
| Beacon speed (°/s) | ≥ 3.5 | 18 | 60.0 | 83 |
| Clouds | yes | 20 | 75.1 | 95 |
| Clouds | no | 20 | 67.9 | 90 |
| Beacon path | circular | 10 | 75.7 | 100 |
| Beacon path | lissajous | 16 | 72.3 | 94 |
| Beacon path | random_walk | 14 | 67.5 | 86 |

## Random scenes, method `hybrid-v2`: time locked (%) by scene property

| Property | Bin | Runs | Time locked mean (%) | Acquired % |
|---|---|---|---|---|
| Turbulence | 0-1 | 10 | 73.5 | 90 |
| Turbulence | 1-1.5 | 16 | 78.1 | 88 |
| Turbulence | ≥ 1.5 | 14 | 63.9 | 100 |
| Decoys | 0-0.5 | 12 | 66.7 | 83 |
| Decoys | 0.5-2.5 | 9 | 82.7 | 100 |
| Decoys | ≥ 2.5 | 19 | 70.2 | 95 |
| Beacon speed (°/s) | 0-2 | 11 | 81.8 | 100 |
| Beacon speed (°/s) | 2-3.5 | 11 | 80.2 | 100 |
| Beacon speed (°/s) | ≥ 3.5 | 18 | 60.9 | 83 |
| Clouds | yes | 20 | 75.2 | 95 |
| Clouds | no | 20 | 68.7 | 90 |
| Beacon path | circular | 10 | 78.5 | 100 |
| Beacon path | lissajous | 16 | 71.8 | 94 |
| Beacon path | random_walk | 14 | 67.5 | 86 |

## Random scenes, method `none`: time locked (%) by scene property

| Property | Bin | Runs | Time locked mean (%) | Acquired % |
|---|---|---|---|---|
| Turbulence | 0-1 | 10 | 52.0 | 60 |
| Turbulence | 1-1.5 | 16 | 66.0 | 75 |
| Turbulence | ≥ 1.5 | 14 | 40.9 | 50 |
| Decoys | 0-0.5 | 12 | 70.5 | 83 |
| Decoys | 0.5-2.5 | 9 | 61.6 | 67 |
| Decoys | ≥ 2.5 | 19 | 39.4 | 47 |
| Beacon speed (°/s) | 0-2 | 11 | 58.7 | 73 |
| Beacon speed (°/s) | 2-3.5 | 11 | 58.2 | 64 |
| Beacon speed (°/s) | ≥ 3.5 | 18 | 47.9 | 56 |
| Clouds | yes | 20 | 50.4 | 60 |
| Clouds | no | 20 | 57.0 | 65 |
| Beacon path | circular | 10 | 51.3 | 60 |
| Beacon path | lissajous | 16 | 58.6 | 62 |
| Beacon path | random_walk | 14 | 49.8 | 64 |
