# Research methodology

MarketForge compares return forecasts under a fixed chronological protocol. A
more complicated model is useful only when it improves a valid out-of-sample
comparison. The selected model is the best development candidate in one run,
not a claim that a model is best for every market or future period.

## Information and execution timing

Each row describes one ticker at the close of session `t`. Features use data
available at or before that close. The target for horizon `h` is:

```text
target[t] = open[t + 1 + h] / open[t + 1] - 1
entry_date = session[t + 1]
exit_date = label_end = session[t + 1 + h]
```

This leaves the next opening auction for execution. It avoids assuming a trader
can observe a closing price and execute at that same price. The final `h + 1`
rows cannot be labeled and are excluded from evaluation. Horizons refer to
observed trading sessions within a ticker, not calendar days.

Rolling features are backward-looking. Rows without complete finite features
are omitted; missing values are not imputed. Scaling, target transforms,
and learned regimes are fit separately inside the training window. A panel split
keeps the same signal date on the same side for every ticker. A training label
must end strictly before the next evaluation segment starts; an additional
configured gap removes nearby signal dates.

This follows the chronological approach described in scikit-learn's
[time-series cross-validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
and its guidance on keeping learned preprocessing inside a
[training-only pipeline](https://scikit-learn.org/stable/common_pitfalls.html).
MarketForge additionally purges by the actual label end date.

## Three separate chronological roles

1. **Research development:** expanding outer windows estimate candidate
   performance. Inner chronological windows choose hyperparameters using only
   each outer training segment. Outer predictions form the development
   leaderboard. The final selected model minimizes development RMSE.
2. **Calibration:** a later, purged development segment estimates the absolute
   residual quantile for the already selected forecast procedure. Calibration
   outcomes do not select models or hyperparameters. Final estimators are fit on
   research data before this segment and are not refit after calibration.
3. **Final holdout:** the last chronological segment measures the frozen
   procedure. Additional candidates' holdout results are diagnostics only and
   never reselect the winner.

The ensemble weights final candidate forecasts by inverse development
mean-squared error. During outer evaluation, only predictions from earlier outer
folds whose labels end strictly before the current fold starts can influence an
ensemble weight; a cold-start fold uses equal weights.
Final weights use all completed research outer predictions. This makes the
ensemble's evaluation distinct from fitting and scoring weights on the same
forecast errors.

The holdout should be consumed once per research decision. Changing the universe,
features, search space, costs, or model after looking at it makes it part of
development. Reserve a new later period for another honest final evaluation.

## Forecast metrics and uncertainty

RMSE is the model-selection objective. MAE, signed directional accuracy, rank
correlation, and R² give complementary diagnostics. A zero forecast is a flat
direction, so it does not receive directional credit for a nonzero return.
Correlation for a constant prediction is reported as zero; R² for a constant
target is undefined. Pooled panel metrics can conceal weak individual assets
or regimes; inspect exported row-level predictions as well.

The paired circular moving-block bootstrap averages squared forecast errors
within each date before sampling consecutive date blocks. This keeps same-date
asset observations together and retains some serial dependence. The reported
95% percentile interval is for `RMSE(selected) - RMSE(reference)`; a negative
value favors the selected model. Dates receive equal weight, so unbalanced
panels can differ from the observation-weighted headline RMSE. Block length is
a sensitivity parameter. The effective length is capped at half the available
dates, rounded down with a minimum of one, so at least two blocks contribute;
a full-sample circular block would create a misleading zero-width interval.
The requested and effective lengths are both recorded. This interval does not
adjust for having tried many candidates.

Prediction interval half-width is the absolute calibration residual at rank
`ceil((n + 1) * (1 - alpha))`. If the rank exceeds the available calibration
sample, no finite width is supported and the function returns infinity. The
experiment runner checks this condition before training and rejects insufficient
calibration samples. Under
exchangeability this is the conventional split-conformal finite-sample rule.
See the primary tutorial by
[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511) for the construction and
its assumptions.
Stock returns are dependent and nonstationary, so nominal coverage is a target,
not a financial-time-series guarantee. Report realized holdout coverage and
average width. A pooled interval also does not guarantee coverage per ticker.

## Trading simulation

The simulator runs only one-session targets. It rejects overlapping or gapped
holding intervals and mixed calendars inside the same signal cohort. Longer
horizons retain forecast diagnostics without pretending their overlapping
returns form an executable daily strategy.

At each signal close, the position direction is the sign of the forecast,
subject to a fixed absolute-return threshold. Each raw position magnitude is
`target_vol / sqrt(252) / trailing_daily_volatility`. The combined gross exposure
is capped at `max_leverage`. This scales exposure using past volatility; it is
not a covariance-aware portfolio volatility target or an optimized strategy.
The forecast magnitude does not otherwise set exposure.

Weights are fractions of current pretrade equity. Turnover is the absolute
change from the previous period's drifted holdings. Flat signals and absent
assets close previous positions. A linear per-dollar cost is paid for initial
entry, every rebalance, and final liquidation. Final liquidation uses holdings
after the last return has accrued. Net returns compound from an initial equity
of one; drawdown includes that initial equity. Non-positive simulated equity is
rejected instead of producing nonsensical compound metrics.

The reference portfolio is equal-weight long, rebalanced each session and
charged the same transaction costs. It is not labeled buy-and-hold. Sharpe uses
daily simple-return mean and sample deviation with 252-session annualization
and no risk-free-rate subtraction. Annualized return likewise assumes daily
sessions. Cost fractions are measured against each period's starting equity;
their sum is a diagnostic, not a dollar P&L reconciliation.

The simulator omits borrow fees, short availability, funding, cash interest,
taxes, price impact, auction capacity, position limits, and partial fills.
Forecast error and simulated profitability should both be reviewed; minimizing
RMSE does not optimize strategy returns.

## Data limitations and reproducibility

Downloaded adjusted OHLCV history is retrospective and can change after vendor
corrections or corporate actions. Price adjustment is useful for continuity but
does not create a point-in-time market database. Preserve the input snapshot and
its hash. A fixed list of currently traded tickers introduces survivorship and
universe-selection bias; research-grade investability needs historical
membership, delisted instruments, and point-in-time corporate actions.

The download sidecar's `sha256` hashes the exact CSV file bytes. A run summary's
`dataset.sha256` hashes the validated, sorted dataframe serialized to CSV. These
serve different purposes and can differ after date or floating-point parsing;
verify each against the corresponding representation rather than comparing
the two strings directly.

Synthetic data verifies the pipeline and provides a controlled demonstration.
Its results are not evidence of predictive skill on real financial data. Run
metadata records the configuration, data source, selected procedure, evaluation
dates, and software details so outputs can be tied to an experiment. Numerical
identity across different hardware or library builds is not guaranteed even
with a fixed seed.
