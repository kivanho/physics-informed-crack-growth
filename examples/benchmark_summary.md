| variant | params | MAE (log) mean [min, max] over 3 outer-split seeds (2026, 2027, 2028) | bias (log) | late / 56 mean [min, max] | fit time [s] | status |
|---|---|---|---|---|---|---|
| physics | 0 | 0.773 [0.773, 0.773] | -0.725 | 7.0 [7, 7] | 0 | VALID |
| constant | 1 | 0.545 [0.536, 0.555] | +0.001 | 30.7 [30, 31] | 0 | VALID |
| ridge | 15 | 0.265 [0.247, 0.279] | -0.005 | 31.0 [30, 32] | 3 | VALID |
| warp_net | 513 | 0.383 [0.349, 0.407] | +0.013 | 29.7 [27, 31] | 137 | VALID |
| decoder | 1280 | 0.314 [0.312, 0.315] | -0.038 | 26.0 [24, 27] | 1650 | VALID |
| decoder_fixed_default | 1008 | 0.382 [0.382, 0.382] | +0.017 | 29.0 [29, 29] | 206 | VALID |
| ablation_no_shape | 1008 | 0.378 [0.378, 0.378] | +0.016 | 29.0 [29, 29] | 226 | VALID |
| ablation_no_ratio | 1008 | 0.382 [0.382, 0.382] | +0.019 | 29.0 [29, 29] | 208 | VALID |
| ablation_no_concavity | 1008 | 0.403 [0.403, 0.403] | +0.011 | 28.0 [28, 28] | 168 | VALID |
| ablation_no_warp | 1008 | 0.378 [0.378, 0.378] | +0.024 | 29.0 [29, 29] | 182 | VALID |
| ablation_no_penalties | 1008 | 0.399 [0.399, 0.399] | +0.015 | 29.0 [29, 29] | 180 | VALID |
| ablation_small_penalties | 1008 | 0.399 [0.399, 0.399] | +0.016 | 29.0 [29, 29] | 214 | VALID |

Initialisation sensitivity (decoder, fixed folds and selections):

- main, init-seed offset 1: MAE 0.347, late 29/56 [VALID]
- main, init-seed offset 2: MAE 0.363, late 33/56 [VALID]
- main, init-seed offset 3: MAE 0.330, late 30/56 [VALID]
