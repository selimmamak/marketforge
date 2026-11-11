# Runtime, installation, and numerical reproducibility

MarketForge supports Python 3.9 and later. The package constraints in
`pyproject.toml` describe supported dependency ranges;
[`requirements-lock.txt`](../requirements-lock.txt) records the package versions
installed for the initial research environment. The lock is a version snapshot,
not a platform-independent, hash-verified environment specification.

## Installing the research environment

From the repository root, create and activate an isolated environment, then
install the project and optional research dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
```

To reproduce the recorded package versions on a compatible Python and platform,
use the lock instead of resolving new dependency versions:

```sh
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

The recorded environment used these principal versions:

| Component | Recorded version |
| --- | --- |
| Python | 3.9.6 |
| NumPy | 2.0.2, linked to Apple Accelerate |
| pandas | 2.3.3 |
| SciPy | 1.13.1 |
| scikit-learn | 1.6.1 |
| PyTorch | 2.8.0 |
| XGBoost | 2.1.4 |
| Platform | macOS 27.2, ARM64 |

Different Python versions, operating systems, processor instructions and native
libraries can change numerical results even with the same Python package pins.
Wheel availability also varies by platform. Use the same input data snapshot,
configuration, seeds, and feature order when comparing runs. Downloading the
same ticker/date range again can return revised provider data. Fetch metadata
records the CSV checksum, and each experiment records its input fingerprint,
configuration, dependency versions, and model warnings.

## XGBoost and OpenMP on macOS

An installed XGBoost wheel may fail to import when the native OpenMP runtime is
missing. The standard macOS installation step is:

```sh
brew install libomp
```

This is the runtime setup described by the
[XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html).
It is separate from installing the Python package.

In the recorded environment, the installed PyTorch distribution already bundled
a compatible OpenMP runtime. Importing PyTorch first allowed XGBoost to import
and complete a small fit/predict check without changing the system. All optional
model families can be launched in that environment with:

```sh
python -c 'import torch; from marketforge.cli import main; raise SystemExit(main())' run \
  --data data/market.csv \
  --config configs/research.yaml \
  --output results/reproduced-study
```

Use a new or empty output directory. This preload is an observed workaround for
the recorded package combination, not a portable replacement for installing
OpenMP. On a correctly configured environment, the ordinary
`marketforge run ...` command needs no preload. The runner records failed model
families explicitly; inspect `model_errors` before interpreting a comparison as
covering every requested model.

## Sampled matrix-operation audit

The recorded NumPy/Accelerate combination issued `divide by zero`, `overflow`,
and `invalid value` warnings from `matmul` despite finite inputs and outputs.
A separate, small numerical diagnostic investigated whether the observed
warnings corresponded to corrupted arithmetic. Compact evidence is preserved in
[`numerics-audit.json`](numerics-audit.json).

The diagnostic first multiplied identity matrices. Sizes 15, 16, 32, 64 and 128
all emitted the three warnings while returning **exactly the correct identity**.
Size 14 returned the exact identity without warnings. Identity multiplication
contains neither division nor values large enough to overflow.

A diagnostic MLP with one 24-unit hidden layer was then fitted for 80 iterations
on 120 rows from the real-data feature matrix and evaluated on another 40 rows.
It used 37 features. Instrumentation checked the actual activation, gradient and
inference matrix products against independent calculations; no forecasting
accuracy claims were derived from this small fit.

| Measurement | Observed result |
| --- | --- |
| Audited matrix products | 402 |
| Products emitting warnings | 321 |
| Individual matrix warnings | 963 |
| Finite outputs | All 402 products |
| Agreement with `np.einsum(..., optimize=False)` | All products at `rtol=1e-12`, `atol=1e-12` |
| Largest absolute difference from `einsum` | 5.6843e-14 |
| Largest absolute difference from scalar `math.fsum` checks | 7.1054e-15 |
| Largest complete forecast difference from an independent `einsum` forward pass | 3.4694e-18 |

The scalar check sampled the first, middle, and last output rows and columns of
each product, up to nine entries per product. The `einsum` comparison checked
every output entry. Disabling `einsum` optimization avoids replacing its
contraction with the same BLAS matrix-product path under investigation.

These observations support the interpretation that the sampled matrix warnings
were spurious floating-point status flags. NumPy documents related interactions
between Apple/ARM streaming matrix instructions, floating-point status flags,
and BLAS warning reporting in
[issue 28687](https://github.com/numpy/numpy/issues/28687) and
[issue 29820](https://github.com/numpy/numpy/issues/29820). Those upstream reports
support the diagnosis; the local numerical comparisons provide the evidence for
this particular environment.

## Limits of the evidence

The audit covers sampled shapes and one small diagnostic network. It does not
prove that every computation in a full research run, a different model, or a
different dependency stack is correct. The independent paths also share
floating-point inputs and are not arbitrary-precision proofs. Retain finite
output checks and inspect warnings in new runs; do not generally suppress
numerical warnings based on this result.

The diagnostic additionally produced a genuine MLP iteration-limit warning.
That warning concerns optimizer convergence and is distinct from the sampled
matrix warning behavior. Fixed iteration budgets make the search reproducible
but do not establish convergence. Neither numerical consistency nor successful
installation establishes forecasting value, profitability, or robustness to a
future market regime.
