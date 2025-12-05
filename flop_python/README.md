# flopsearch

Python package providing an implementation of the FLOP causal discovery algorithm for linear additive noise models.

## Installation
flopsearch can be installed via pip:

```bash
pip install flopsearch
```

## Citing FLOP
If you use FLOP in your scientific work, please cite this paper:
```bibtex
@article{cifly2025,
  author  = {Marcel Wien{"{o}}bst and Leonard Henckel and Sebastian Weichwald},
  title   = {{Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning}},
  journal = {{arXiv preprint arXiv:2510.04970}},
  year    = {2025}
}
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
flopsearch.flop(X_std, 2.0, restarts=50)
```

## Input and Output
As input, FLOP takes the data matrix, the BIC penalty parameter (we recommend ```2.0``` as a default choice) and either a ```timeout``` (in seconds) or the number of ILS restarts to control how long the search runs.

The output of FLOP is a CPDAG encoded with an adjacency matrix whose entry in row i and column j is 1 in case of a directed edge from the i-th to the j-th variable and 2 in case of an undirected edge between those variables (in case of an undirected edge, the entry in row j and column i is also 2, that is each undirected edge induce two 2's in the matrix).
