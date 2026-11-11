# Model library and computational design

MarketForge predicts a numerical forward return from a panel of engineered
market features. Every estimator receives exactly the same ordered feature
columns and fold boundaries. Features are observable at the signal timestamp;
preprocessing learned from data is fitted inside each training fold.

The library deliberately includes simple reference forecasts. Complexity is
useful only when it improves a genuinely later evaluation period. A baseline
winning the leaderboard is a valid result.

## Estimators

| Registry name | Method | What it contributes |
| --- | --- | --- |
| `zero` | Predict a zero return | No-change reference with no fitted predictive signal |
| `historical_mean` | Training-sample average return | A drift reference fitted only to available training labels |
| `ridge` | L2-regularized linear regression | A stable linear combination of correlated technical features |
| `elastic_net` | Combined L1/L2-regularized linear regression | Sparse coefficients with stabilization for correlated predictors |
| `random_forest` | Bagged randomized regression trees | Nonlinear interactions and threshold effects with leaf-size regularization |
| `extra_trees` | Randomized tree thresholds | An alternative bias/variance tradeoff to the random forest |
| `hist_gradient_boosting` | Histogram-based gradient boosting | Sequential fitting of residual errors with constrained tree size |
| `svr` | Radial-basis support vector regression | Smooth nonlinear responses in standardized feature space |
| `mlp` | Multilayer feed-forward network | Learned nonlinear feature interactions with L2 regularization |
| `regime_mixture` | Gaussian soft gate and local ridge experts | Different predictive relationships in different feature distributions |
| `xgboost` | Optional regularized boosted trees | Row/column subsampling, shrinkage, and explicit tree/weight regularization |
| `torch_attention` | Optional feature-token Transformer | Attention-based interactions among features of one observation |
| `torch_residual` | Optional gated residual neural network | Adaptive nonlinear corrections around learned feature representations |

`ridge`, `elastic_net`, `svr`, and `mlp` use a scikit-learn pipeline with feature
standardization and a `TransformedTargetRegressor` that standardizes training
targets. Predictions are transformed back to return units. Tree models consume
the unscaled finite feature matrix. Their training iterations are fixed; they
do not create randomized validation holdbacks for early stopping.

## Regime mixture of experts

The custom regime model uses the following training procedure:

1. Standardize features using only the supplied training rows.
2. Project the standardized matrix onto at most six principal components.
3. Fit a regularized, full-covariance Gaussian mixture in that reduced space.
4. Fit one global ridge model and one ridge expert per Gaussian component. The
   local expert uses each training row's posterior component probability as its
   sample weight.
5. Shrink each local prediction toward the global prediction according to the
   expert's effective sample size.

For component responsibilities \(w_{ik}\), the effective sample size is

\[
n_{\mathrm{eff},k}=\frac{(\sum_i w_{ik})^2}{\sum_i w_{ik}^2},
\qquad
s_k=\frac{n_{\mathrm{eff},k}}{n_{\mathrm{eff},k}+\lambda_s}.
\]

An unseen observation receives forecast

\[
\hat y(x)=\sum_k p(k\mid x)
\left[s_k\hat y_k(x)+(1-s_k)\hat y_{\mathrm{global}}(x)\right].
\]

The gate remains frozen during inference. Small or constant training sets reduce
to one component. The components identify statistical feature regimes; the
implementation does not assign them verified economic labels such as recession,
bull market, or crisis. Full covariances preserve correlations after PCA rotation.

## Neural architectures

The feature-attention model maps each scalar feature to a learned vector using
feature-specific weights and biases. A learned summary token and two
four-head Transformer encoder layers combine those tokens, followed by layer
normalization and a scalar prediction head. This is a **tabular feature-token
Transformer**, not a temporal Transformer. Adjacent rows may belong to different
tickers and are never interpreted as successive time steps. Historical
information reaches the model through the causal lag and rolling features.

The residual network first projects the feature vector into a hidden space. Two
blocks then apply layer normalization, a two-layer GELU transformation, dropout,
and a learned sigmoid gate before adding the correction to the block input.
Layer normalization and a final linear head produce the normalized forecast.

Both neural models use training-only feature and target normalization, AdamW,
Huber loss, gradient clipping, fixed epoch budgets, and CPU execution. A seeded
PyTorch random-state context makes repeated fits reproducible on the same
software/hardware stack without permanently changing the caller's RNG state.
Training-loss history is saved on the estimator as `loss_history_`; it is not an
out-of-sample accuracy measure. Neither neural model implements an LSTM or a
temporal sequence model.

## Search and temporal ensembling

`model_candidates` returns an explicit finite parameter grid. The normal grid
varies regularization, tree complexity, regime count, hidden width and training
budget. Quick mode uses one reduced-cost candidate per family, so it evaluates
model families without inner parameter selection.

The CPU research grid for `torch_attention` uses 16 hidden dimensions with
12 or 20 epochs. The `torch_residual` grid compares 32 hidden dimensions with
20 epochs against 64 hidden dimensions with 30 epochs. Quick mode uses 12 epochs
and widths of 16 and 32, respectively. These bounded budgets make nested
evaluation practical; they are not evidence that a neural model has converged.

The experiment runner selects parameters on purged inner time folds and scores
each family on outer walk-forward folds. Its temporal ensemble uses inverse
mean-squared-error weights estimated from earlier completed outer folds. The
first outer fold uses equal weights. Final ensemble weights are learned from
the development predictions and then frozen. Calibration and final holdout
labels are not used to select the winner or update ensemble weights. See the
methodology document and the experiment implementation for exact boundaries.

## Runtime and dependencies

Let \(N\) be training rows, \(F\) feature count, \(K\) regime count, \(D\) gate
dimension, \(T\) tree count, and \(E\) neural epochs. Dense linear fits generally
scale with approximately \(NF^2\); trees add repeated split searches over rows
and features. RBF support vector regression can approach quadratic memory and
super-quadratic fitting time as \(N\) grows. Gating incurs Gaussian-mixture
iterations with approximately \(NKD^2\) work per iteration, followed by \(K\)
weighted linear fits. Feature attention contains a term quadratic in \(F\),
while the residual network scales linearly in \(F\) before its hidden layers.
All these fitting costs are multiplied by the number of candidates and time
folds; there is no free gain from increasing search size.

Run quick mode first to validate data and estimate runtime, then use the normal
grid for parameter selection. The registry uses one CPU thread for tree models;
the runner also limits supported native thread pools during fitting. Actual
runtime depends on the platform, sample count, and BLAS/OpenMP configuration.

Install optional packages with `pip install '.[neural,boosting]'`. XGBoost also
requires an OpenMP runtime, such as `libomp` on macOS. A package being installed
does not prove its native library can load: the runner records unavailable
models rather than inventing scores. The neural module is imported only when a
PyTorch model is constructed.

## References

- [scikit-learn GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [scikit-learn leakage and preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html)
- [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
