# Dataset 31 Target Variation — Task 1 Audit

## Current Benchmark (1891 examples, 70% train 1323 / 30% test 568, trajectory-level first 70% time → last 30% time, win 1024, E=1, 56Msps 3.4GHz)
- **Number of unique captures/windows:** 1323 train / 568 test, each window reads different real region of `a2a.sigmf-data` via `sigmf_sample_start` 0,56000,112000...204120000 etc., **verified unique** (1323/1323, 568/568).
- **Range:** train 18.20/22.09/19.97/0.94 (min/max/mean/std), test 18.45/21.92/20.11/0.79, all 18.20/22.09/19.97/0.90 — **hist train [256,359,286,295,127] test [68,160,149,138,53] bins 18.2-22.0**
- **Radial velocity:** train -6.09/6.12/0.018/2.33, test -5.73/5.74/0.015/1.99, all -6.09/6.12/0.018/2.23 — **hist train [112,122,847,127,115] test [40,55,368,73,32]**
- **3D velocity:** train min [-2.17,-5.77,-0.30] max [2.26,5.77,0.39] mean [-0.014,0.038,-0.003] std [1.28,3.46,0.11], test min [-2.26,-5.77,-1.50] max [2.71,5.77,1.49] mean [-0.028,0.129,-0.005] std [1.30,3.50,0.15]
- **Windows per trajectory:** single flight spherical, 1891 measurements, 8883 SigMF captures (56k each, 1ms), csv time_s dt 0.1s, SigMF 20ms/100ms, sample_start 0,56000,112000...204M
- **Temporal spacing:** 0.1s (csv) and 1ms (SigMF), 4-window temporal = 0.4s total
- **Source IQ sample_start:** train first 3 [0,56000,112000], test first 3 [204120000,204232000,204288000] — **every sample reads different real region** (verified via memmap hash fba986 vs dummy 1f6a)
- **Rejection:** **REJECT current benchmark for range** — target variation too small (std 0.90, range 18-22, baseline MAE 0.67) to distinguish learning from predicting mean (test mean 20.11 vs train mean 19.97 diff 0.14 < std). **Radial velocity variation sufficient** (std 2.23, range -6 to 6, baseline MAE 1.22) — can distinguish. 3D velocity also sufficient.

