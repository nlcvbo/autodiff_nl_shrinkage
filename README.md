# autodiff non-linear shrinkage

The code is linked to the article [Asymptotic non-linear shrinkage and eigenvector overlap for weighted sample covariance](https://arxiv.org/abs/2410.14420).

Non-linear shrinkage estimation of covariance and precision matrices for
**weighted** samples, where the population spectrum is identified by automatic
differentiation: simulated sample spectra are matched to the observed one by
gradient descent (PyTorch), and the fitted population spectrum is then plugged
into the non-linear shrinkage formulas. 

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and requires Python >= 3.12.

```bash
uv sync
```

## Quick start

```python
import numpy as np
import torch

from autodiff import Autodiff
from utils import make_weight_distribution, make_weights

p, n = 100, 1000
tau_pop = np.array([1.0] * 20 + [3.0] * 40 + [10.0] * 40)

# Exponentially decaying sample weights
d, wd = make_weight_distribution()
W = make_weights("ewma", n, 5.0, d, wd)

X = np.random.normal(size=(n, p)) * np.sqrt(tau_pop)
S = X.T @ (W[:, None] * X) / n

estimator = Autodiff(bias=True, assume_centered=True).fit_from_S(S, W, n_epochs=50, lr=5e-2)
covariance = estimator.get_covariance(d=d, wd=wd, weights="ewma", w_args=[5.0, 30])
precision = estimator.get_precision(d=d, wd=wd, weights="ewma", w_args=[5.0, 30])
tau_hat = estimator.get_tau()  # estimated population eigenvalues
```

`weights` is one of `"ewma"` (`w_args=[alpha, Kt]`), `"unif"` (`w_args=[alpha, Kt]`)
or `"Ndiracs"` (`w_args=[Kd, Kt]`, uses the discrete weight law `(d, wd)`).

## Experiments

From the `src` directory:

```bash
uv run main.py [scale] [n_jobs]
```

runs covariance and precision estimation (loss and PRIAL against the sample,
linear-shrinkage and oracle non-linear estimators), then the two robustness
experiments below with `scale=0.05` by default. They can also be run alone:

```bash
uv run heavy_experiments.py [scale] [n_jobs]
uv run spike_experiments.py [scale] [n_jobs]
```

* `heavy_experiments.py`: concentration ratio `c` near 1 and `c > 1`, singular
  population covariance, heavy tails (Student, Pareto, elliptical noise; true
  and retrieved spectrum). Summaries are written to `heavy_results.json`.
* `spike_experiments.py`: location and shrinkage of the outliers created by
  population spikes. Summaries are written to `spike_results.json`.

`scale` multiplies the numbers of Monte-Carlo runs (`1` reproduces the article,
e.g. `0.05` for a quick check) and `n_jobs` is the number of worker processes.
With the default sizes the full sets take hours.

## Layout

| File | Content |
| --- | --- |
| `autodiff.py` | `Autodiff`, the scikit-learn compatible estimator |
| `utils.py` | differentiable sampler, losses, optimisation loop and sample weights |
| `nl_formulas.py` | oracle non-linear shrinkage of covariance / precision eigenvalues |
| `nl_utils.py` | Stieltjes transform solvers and asymptotic sample density |
| `weighted_LWO_estimator.py` | weighted Ledoit-Wolf-type linear shrinkage |
| `support/` | support identification (per weight family) and root finding, only used to plot the asymptotic density |
| `main.py` | covariance and precision estimation experiment; runs all experiments |
| `heavy_experiments.py`, `heavy_utils.py` | heavy-tail experiments and the Monte-Carlo machinery shared with the spike experiments |
| `spike_experiments.py`, `spike_utils.py` | spike experiments and their helpers |

## Development

```bash
uvx ruff check .    # lint
uvx ruff format .   # format
```
