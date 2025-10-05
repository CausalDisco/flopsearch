# flopsearch

Python package providing an implementation of the FLOP causal discovery algorithm for linear additive noise models.

## Installation
flopsearch can be installed via pip:

```bash
pip install flopsearch
```

## Example
A simple example run of the FLOP algorithm provided by flopsearch.

``` py
import flopsearch
import numpy as np
from scipy import linalg

p = 10
W = np.diag(np.ones(p - 1), 1)
X = np.random.randn(10000, p).dot(linalg.inv(np.eye(p) - W))
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
flopsearch.flop(X_std, 2.0, restarts=20)
```
