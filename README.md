# FLOP Causal Discovery Algorithm

This repository contains a Rust implementation of the [FLOP causal discovery algorithm](https://arxiv.org/abs/2510.04970), available for use from Python and R. It is a score-based algorithm for learning equivalence classes of DAGs from observational data, assuming linear relationships between variables. 

## Installation
In Python, flopsearch can be installed via pip:

```bash
pip install flopsearch
```

In R, flopsearch can be installed directly from GitHub:

``` r
install.packages("https://github.com/CausalDisco/flopsearch/releases/download/v0.2.1/flopsearch.tar.gz")
```

This requires a working installation of the [Rust toolchain](https://rust-lang.org/tools/install/).

The name of the installed package is ```flopsearch```, and it can be loaded with:

``` r
library(flopsearch)
```

## Citing FLOP
If you use FLOP in your scientific work, please cite this paper:
```bibtex
@article{embracing2025,
  author  = {Marcel Wien{\"o}bst and Leonard Henckel and Sebastian Weichwald},
  title   = {{Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning}},
  journal = {{arXiv preprint arXiv:2510.04970}},
  year    = {2025}
}
```

## How To Run FLOP
In Python, as a simple example, FLOP can be called by
``` py
flopsearch.flop(X, 2.0, restarts=50)
```
with ```X``` being the data matrix, ```2.0``` the BIC penalty parameter and the number of ILS ```restarts``` being set to ```50```.

Similarly, in R, one can call:
``` r
flopsearch::flop(X, 2.0, restarts=50)
```

Instead of the number of restarts, it is also possible to set a ```timeout``` in seconds after which the search terminates and returns the best-scoring graph found thus far.

FLOP returns a CPDAG encoded with an adjacency matrix whose entry in row i and column j is 1 in case of a directed edge from the i-th to the j-th variable and 2 in case of an undirected edge between those variables (in case of an undirected edge, the entry in row j and column i is also 2, that is each undirected edge induces two 2's in the matrix).
