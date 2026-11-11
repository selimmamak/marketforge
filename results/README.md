# Initial Forecasting Study

Completed November 10, 2026. Random Forest had the lowest development error, but
it did **not** improve on the zero-return baseline in the final holdout. This
study does not establish a forecasting or trading edge.

## Data and experiment

- Universe: AAPL, MSFT, QQQ, SPY.
- Price history: January 4, 2016 through November 10, 2026. There are 10,920 daily OHLCV records.
- Features: 37 causal market features, producing 10,672 labeled asset observations.
- Models: 13 model families plus the temporal ensemble. Every requested family completed successfully.
- Selection: three purged outer windows, with two inner windows for hyperparameter search. The full grids in [research.yaml](../configs/research.yaml) were used.
- Final model training ends February 2, 2026. Separate interval calibration runs from February 5 through May 6.
- Holdout signals: May 11 through November 6, 2026. Their open-to-open returns finish on November 10, covering 126 sessions and 504 asset observations.

The winning model was frozen before inspecting the holdout. Other models'
holdout results are diagnostics and did not change that selection.

## Main results

RMSE is measured in simple-return decimal units. Lower is better.

| Measure | Selected Random Forest | Reference |
| --- | ---: | ---: |
| Development RMSE | 0.011745 | 0.011835, zero-return forecast |
| Holdout RMSE | 0.017093 | 0.016829, zero-return forecast |
| Holdout MAE | 0.011623 | 0.011527, zero-return forecast |
| Holdout directional accuracy | 52.58% | A zero forecast represents a flat direction |
| Simulated holdout total return | -0.89% | +12.24%, equal-weight long portfolio |
| Maximum drawdown | -11.31% | -10.67%, equal-weight long portfolio |

The trading reference rebalances equally across the four assets each session.
Both portfolios include the same cost assumption of 5 basis points per dollar
traded, including entry, rebalancing, and final liquidation. This is a simulated
comparison with idealized open fills, not an executed account history.

The selected model's holdout RMSE minus the zero-return reference is 0.000264.
Its paired, date-clustered block-bootstrap 95% interval is
**[-0.000167, 0.000780]**. It crosses zero and does not demonstrate a reliable
error reduction. The nominal 90% prediction interval covered **87.10%** of
holdout observations; its average width was 0.048715 in return units.

No search settings were changed in response to these final holdout outcomes.
The historical-mean baseline had the lowest diagnostic holdout RMSE, but it was
not retrospectively substituted for the development-selected model. A new
research decision needs a new future evaluation period.

## Saved artifacts

| File | Purpose |
| --- | --- |
| [Report](initial-study/report.html) | Self-contained HTML report. Download or open locally in a browser. |
| [Run summary](initial-study/summary.json) | Complete metrics, date boundaries, configuration, hashes, versions, and fit warnings. |
| [Development leaderboard](initial-study/leaderboard.csv) | All 14 candidates ranked using development predictions. |
| [Holdout metrics](initial-study/holdout_metrics.csv) | Diagnostic final-period metrics for every candidate. |
| [Development predictions](initial-study/development_predictions.csv) | Per-row out-of-fold predictions and fold identifiers. |
| [All holdout predictions](initial-study/all_holdout_predictions.csv) | Per-row forecasts for every candidate. |
| [Selected holdout predictions](initial-study/holdout_predictions.csv) | Random Forest forecasts, outcomes, and intervals. |
| [Daily strategy returns](initial-study/daily_returns.csv) | Costs, turnover, exposure, equity, and benchmark returns. |
| [Tuning audit](initial-study/tuning.json) | Inner validation trials and chosen parameters. |
| [Selected model](initial-study/model.joblib) | Compressed Random Forest checkpoint with feature order and interval calibration. |
| [Latest forecasts](initial-study/latest_forecasts.csv) | November 10 forecasts from the same frozen research model. |

The latest forecasts retain the February 2 training cutoff. They are research
outputs, not forecasts from a model retrained on the final holdout. The artifact
contains only the selected model, so inference does not require the optional
neural or boosting packages. Only load trusted joblib files.

## Reproduce or resume

Install the dependencies described in the [runtime notes](../docs/runtime.md).
Run from the repository root with a fresh output directory:

```bash
marketforge run --data data/market.csv --config configs/research.yaml \
  --output runs/reproduced-study --workers 3
```

On the recorded Mac, the documented PyTorch preload supplies the OpenMP runtime:

```bash
python -c 'import torch; from marketforge.cli import main; raise SystemExit(main())' run \
  --data data/market.csv --config configs/research.yaml \
  --output runs/reproduced-study --workers 3
```

After an interruption, repeat the same command with `--resume`. Verified local
checkpoints reuse completed fits. They are excluded from Git. The committed
selected-model checkpoint can be used directly:

```bash
marketforge forecast --data data/market.csv \
  --artifact results/initial-study/model.joblib \
  --output runs/latest-forecasts.csv
```

The completed run used three workers and took approximately 384 seconds on the
recorded environment. Runtime and exact floating-point results can vary by
hardware and package build. See [methodology](../docs/methodology.md) for timing,
data limitations, and uncertainty assumptions, and the [numerical audit](../docs/runtime.md)
for the observed local math-library warnings.
