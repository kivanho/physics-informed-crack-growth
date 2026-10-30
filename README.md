# Physics-informed crack growth

Synthetic-data demonstration of a physics-informed knot decoder for short-crack growth, benchmarked against simpler corrections of the same physics model on identical cross-validation folds.

[![CI](https://img.shields.io/badge/CI-tests%20%2B%20smoke%20benchmark-142E46)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-142E46)](requirements.lock)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-087E85)](requirements.lock)
[![License: MIT](https://img.shields.io/badge/license-MIT-4F6F94)](LICENSE)

## 1. Engineering question

A fracture-mechanics model predicts the number of cycles until a defect's short crack reaches the short-to-long-crack transition (the *cycles to crack nucleation*). On real specimens the model is directionally right but biased. **Can a small network correct that bias from static specimen descriptors while the predicted crack-growth curves stay physically plausible, and is the curve-shaped decoder worth its complexity compared with simply rescaling the physics prediction?**

On the synthetic data in this repository the second half is answered negatively (section 5). That negative result is kept deliberately.

## 2. What the repository implements

- **Knot decoder** ([`picg/decoder.py`](picg/decoder.py)): an MLP maps specimen descriptors to 16 numbers: 7 + 7 positive density knots (softplus), a bounded terminal `c/a` ratio and a bounded log cycle warp. A shape-preserving cubic (PCHIP) through the knots is integrated to monotone crack-extension curves for the depth and surface fronts; a cyclic R-curve maps extension to threshold; the first crossing of the transition threshold is the predicted cycle count `N = N_phys · exp(warp)`. Curves are monotone and positive by construction. The PCHIP slopes are finite in the forward *and* backward pass (constant or pairwise-equal knots used to give `nan` gradients; see [`tests/test_decoder_gradients.py`](tests/test_decoder_gradients.py)).
- **Physics-informed loss** ([`picg/losses.py`](picg/losses.py)): MSE of log cycles plus four penalties against the physics reference curves: Wasserstein-1 shape distance, Huber terminal-ratio error, excess convexity of the threshold curve, and warp size. A weight of exactly `0` removes a term from the graph (true ablation); a small weight keeps it active.
- **Leakage-safe nested cross-validation** ([`picg/cv.py`](picg/cv.py), [`picg/model.py`](picg/model.py)): scaling, encoding, loss-weight profile, backbone and epoch count are all chosen inside each outer training set. One training-budget strategy everywhere: fixed epoch budget, epoch count chosen by the mean inner-validation MAE, final fit with that count, no early stopping on the training objective.
- **Baselines on identical folds** ([`picg/baselines.py`](picg/baselines.py), [`picg/benchmark.py`](picg/benchmark.py)): the physics model; one multiplicative correction; ridge regression from features to the log correction; a warp-only network (the decoder's timing head without curve heads); the full decoder; seven fixed-configuration decoder ablations.
- **Strict evaluation** ([`picg/metrics.py`](picg/metrics.py)): a non-finite or non-positive prediction makes the whole set `INVALID`; metrics are never computed over a subset, every count reports its denominator, and affected specimen ids are listed. Training raises `NumericalError` (epoch, step, loss term, parameter) on any non-finite loss or gradient.
- **Run artifacts**: manifest (commit, environment, data fingerprint, configuration, validity), fold and inner-split assignments, per-specimen predictions for every variant, inner-search tables, per-epoch training histories.

## 3. Data provenance and the public/private boundary

Everything here is **synthetic**. [`picg/physics.py`](picg/physics.py) is a textbook stand-in (cyclic R-curve threshold + Paris-type growth with simplified geometry factors, illustrative constants). [`picg/data.py`](picg/data.py) draws a 56-specimen campaign and sets the "observed" cycles to the physics prediction times a smooth, hidden, feature-dependent bias times log-normal scatter (σ = 0.30 in log cycles, a noise floor of roughly 0.24 MAE(log)).

The method comes from my M.Sc. thesis with an industrial partner. The experimental data and the partner's physics implementation are **not** included, and nothing in this repository reproduces or approximates the thesis results. The synthetic task is easier than the real one (the hidden bias is a smooth function of the inputs), so no number below says anything about real-world accuracy.

## 4. Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate     # or: python3.12 -m venv .venv
pip install -r requirements.lock                             # pinned; the tests and benchmark ran from it
pytest -q                                                    # about 1.5 min, 66 tests
python -m picg.benchmark --smoke --out runs/smoke            # about 1 min end-to-end check (what CI runs)
python -m picg.run                                           # quick demo preset (physics, constant, ridge, decoder)
python -m picg.benchmark --out runs/main --init-seeds 1 2 3  # full benchmark; timing in section 5
```

`runs/` is gitignored; each run directory holds `manifest.json`, `folds.json`, `predictions.csv`, `metrics.json`, `inner_search.csv`, `histories.json` and `summary.md`. Use your own table with `--table my_specimens.csv` (columns as in [`examples/synthetic_specimens.csv`](examples/synthetic_specimens.csv)).

## 5. Main evidence

<!-- RESULTS -->

## 6. Limitations and unsupported uses

- **Synthetic stand-ins only.** Physics model, constants and specimens are illustrative. Curve "fidelity" is agreement with the stand-in physics curves, not with measured crack-growth curves.
- **The decoder's timing is one scalar.** The depth front reaches the transition at `s = 1` by construction, so the predicted cycle count is `N_phys · exp(warp)` (earlier only when the surface front is scaled above the depth front). The 14 knots shape the curves; they do not move the crossing. This is why a ridge regression on the log correction is a fair competitor.
- **Non-conservative predictions.** Every learned correction removes the physics model's conservative bias and leaves a large share of blind predictions *late* (more cycles predicted than observed). None of these models is a design tool; a safety margin or an explicitly conservative objective would be required, and neither is implemented.
- **No prediction intervals.** Point predictions only. The bootstrap interval in section 5 is the sampling uncertainty of a *metric* over 56 blind specimens with the models fixed, not a per-specimen uncertainty.
- **Small data, few repeats.** 56 specimens, 3 outer-split seeds, 4 initialisation seeds. Spread across outer-split seeds mixes fold composition with optimisation randomness; the inner splits overlap and are not independent replicates.
- **Epochs, not gradient steps, are transferred** from the inner fits (about two thirds of the outer training set) to the final fit.
- **Post-transition tail** `s ∈ (1, 1.2]` is unsupervised; do not interpret it.
- **Determinism** is best-effort: seeds are fixed and TensorFlow op determinism is enabled, but results can differ across platforms and TensorFlow releases.

## 7. Tests and reproducibility

- `pytest -q`: 66 tests covering PCHIP finiteness and gradients (constant/equal knots, flat segments, sign changes, tiny slopes, finite differences, gradients through the real network), decoder monotonicity, the crossing function, the concavity penalty, split hygiene, strict metrics (NaN/inf/non-positive/shape/empty/complete coverage), training-loop checks, true ablations, baseline leakage, and a tiny end-to-end benchmark that must end `VALID`.
- [`requirements.lock`](requirements.lock): pinned universal lock compiled with `uv pip compile requirements.in -o requirements.lock --python-version 3.12 --universal`. The tests and the benchmark in section 5 ran from it (Python 3.12.14, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1), and the quick start was repeated in a fresh virtual environment.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): installs the lock, runs the tests and the smoke benchmark (exit 1 on numerical failure, exit 2 on `INVALID` coverage); the full benchmark is a `workflow_dispatch` option.
- Every run writes a manifest with commit hash, dirty flag, package versions, platform, the SHA-256 of the specimen table, the data seed, the full configuration and the validity status, next to fold assignments, per-specimen predictions, inner-search results and per-epoch histories. The committed [`examples/benchmark_summary.json`](examples/benchmark_summary.json) and [`examples/synthetic_blind_predictions.csv`](examples/synthetic_blind_predictions.csv) are the small public outputs the tables above are read from.

## Repository layout

| Path | Contents |
|---|---|
| `picg/physics.py` | synthetic short-crack model |
| `picg/data.py` | synthetic specimen generator, event grid, physics reference curves |
| `picg/decoder.py` | 16-parameter knot decoder, safe PCHIP, network |
| `picg/losses.py` | first-crossing function, log-cycle loss, four physics penalties |
| `picg/cv.py` | outer/inner splits, leakage-safe preprocessing |
| `picg/model.py` | fixed-budget training loop with finite checks, inner search, nested CV |
| `picg/baselines.py` | physics, constant, ridge, warp-only network, decoder, ablations |
| `picg/benchmark.py` | shared-fold benchmark runner, artifacts, `--summarize` |
| `picg/metrics.py` | strict metrics with validity status and denominators |
| `picg/manifest.py` | run manifest helpers |
| `picg/run.py` | quick demo preset |
| `scripts/make_figures.py` | README figure |
| `tests/` | 66 tests |
| `examples/` | synthetic table, blind predictions, benchmark summary |

## Citation / context

Method from the M.Sc. thesis *Physics-Informed Machine Learning for Modeling Short Crack Growth in Gas Turbine Components with Manufacturing Defects* (Kien van Ho, 2026), written with an industrial partner. This repository is an independent re-implementation and benchmark on synthetic data.

```bibtex
@software{vanho2026picg,
  author = {van Ho, Kien},
  title  = {Physics-informed crack growth: a knot decoder for short-crack growth (synthetic demo)},
  year   = {2026},
  url    = {https://github.com/kivanho/physics-informed-crack-growth}
}
```

More projects: [kivanho.github.io](https://kivanho.github.io). MIT licence, see [LICENSE](LICENSE).
