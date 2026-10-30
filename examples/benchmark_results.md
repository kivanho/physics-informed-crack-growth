All numbers below were produced by `python -m picg.benchmark` at commit `fbd95d5d30` from the pinned environment; status **VALID** (56/56 blind predictions valid for every variant, 0 specimens excluded before evaluation). Configuration: 56 synthetic specimens (data seed 2026), 5 outer folds (split seed 2026), 3 inner splits per fold, fixed budget 100 epochs with the epoch count chosen on the inner splits, decoder search over 4 loss profiles x 2 backbones. Metrics are on `log1p` cycles over all 56 blind specimens (every specimen is blind exactly once). *Late* = predicted more cycles than observed (non-conservative). W1 = Wasserstein-1 distance of the decoded, normalised extension curves to the **synthetic physics reference curves** (agreement with the prior, not accuracy against truth); variants without curve heads use the physics curves unchanged. Fit time is the summed wall time over the 5 folds (including inner search) while three benchmark processes shared one 8-core laptop CPU, so absolute times are inflated.

### Baselines and decoder on identical folds (split seed 2026)

| variant | params | MAE (log) | RMSE (log) | bias (log) | late / 56 | MAE minus physics MAE, 95 % specimen bootstrap | W1 to physics curves | fit time [s] |
|---|---|---|---|---|---|---|---|---|
| 1. physics model | 0 | 0.773 | 0.974 | -0.725 | 7 | reference | physics curves | 0 |
| 2. constant correction | 1 | 0.543 | 0.663 | +0.002 | 30 | -0.230 [-0.378, -0.077] | physics curves | 0 |
| 3. ridge log-correction | 15 | 0.279 | 0.356 | +0.001 | 31 | -0.494 [-0.661, -0.344] | physics curves | 3 |
| 4. warp-only network | 513 | 0.349 | 0.470 | +0.039 | 31 | -0.424 [-0.610, -0.247] | physics curves | 115 |
| 5. full decoder (inner search) | 1280 | 0.314 | 0.409 | -0.022 | 27 | -0.459 [-0.623, -0.311] | 0.0773 | 1512 |

### Decoder ablations (fixed 32-unit backbone, default weights unless stated; epochs chosen on the inner splits)

| configuration | params | MAE (log) | bias (log) | late / 56 | W1 to physics curves | excess convexity | terminal-ratio MAE vs physics | fit time [s] |
|---|---|---|---|---|---|---|---|---|
| decoder, default weights, no search | 1008 | 0.382 | +0.017 | 29 | 0.0724 | 3.42e-02 | 0.0590 | 206 |
| all penalties small but active (1e-2 / 1e-3) | 1008 | 0.399 | +0.016 | 29 | 0.0743 | 6.28e-02 | 0.0704 | 214 |
| shape penalty disabled (w = 0) | 1008 | 0.378 | +0.016 | 29 | 0.0836 | 3.54e-02 | 0.0601 | 226 |
| ratio penalty disabled (w = 0) | 1008 | 0.382 | +0.019 | 29 | 0.0719 | 3.18e-02 | 0.0667 | 208 |
| concavity penalty disabled (w = 0) | 1008 | 0.403 | +0.011 | 28 | 0.0490 | 2.08e+00 | 0.0440 | 168 |
| warp penalty disabled (w = 0) | 1008 | 0.378 | +0.024 | 29 | 0.0722 | 3.25e-02 | 0.0594 | 182 |
| all four penalties disabled (w = 0) | 1008 | 0.399 | +0.015 | 29 | 0.0781 | 4.67e-01 | 0.0776 | 180 |

Inner-search selections for the full decoder: fold 1: [32, 16] units, w_shape 1, w_ratio 1, w_concav 1, w_warp 0.1, 96 epochs, fold 2: [32, 16] units, w_shape 1.14, w_ratio 2.88, w_concav 7.21, w_warp 0.747, 71 epochs, fold 3: [32] units, w_shape 0.01, w_ratio 0.01, w_concav 0.01, w_warp 0.001, 39 epochs, fold 4: [32, 16] units, w_shape 1.14, w_ratio 2.88, w_concav 7.21, w_warp 0.747, 67 epochs, fold 5: [32, 16] units, w_shape 1, w_ratio 1, w_concav 1, w_warp 0.1, 82 epochs.

### Variability: 3 outer-split seeds (2026, 2027, 2028) and 4 initialisation seeds

Three kinds of spread are reported separately and none of them is a prediction interval:

| variant | MAE (log) mean [min, max] over split seeds | late / 56 mean [min, max] |
|---|---|---|
| 1. physics model | 0.773 [0.773, 0.773] | 7.0 [7, 7] |
| 2. constant correction | 0.545 [0.536, 0.555] | 30.7 [30, 31] |
| 3. ridge log-correction | 0.265 [0.247, 0.279] | 31.0 [30, 32] |
| 4. warp-only network | 0.383 [0.349, 0.407] | 29.7 [27, 31] |
| 5. full decoder (inner search) | 0.314 [0.312, 0.315] | 26.0 [24, 27] |

- **Split variability (table above):** repeating the whole benchmark with other outer folds. The spread mixes fold composition with the optimisation randomness of retraining, so it is an upper bound on either alone.
- **Optimisation variability (split seed 2026, folds and selected configurations fixed, only the final-fit initialisation and batch order changed):** decoder MAE(log) 0.314, 0.347, 0.363, 0.330 (range 0.314 to 0.363); late 27, 29, 33, 30 / 56.
- **Statistical uncertainty of the metric (models fixed, 2000 paired bootstrap resamples of the 56 blind specimens, split seed 2026):** decoder MAE minus physics MAE -0.459, 95 % interval [-0.623, -0.311]; ridge minus physics -0.494 [-0.661, -0.344].
