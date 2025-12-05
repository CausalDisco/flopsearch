# FLOP Causal Discovery Algorithm

This repository contains a Rust implementation of the FLOP causal discovery algorithm available from Python and R. 

## Installation
In Python, flopsearch can be installed via pip:

```bash
pip install flopsearch
```

In R, flopsearch can be installed directly from Github:

``` r
install.packages("https://github.com/CausalDisco/flopsearch/releases/download/v0.1.3/flopsearch.tar.gz")
```

This requires a working installation of the [Rust toolchain](https://rust-lang.org/tools/install/).

The name of the installed package is ```flopsearch``` and it can be loaded with:

``` r
library(flopsearch)
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

## How To Run FLOP
In Python, as a simple example, FLOP can be called by
``` py
flopsearch.flop(X, 2.0, restarts=50)
```
with ```X``` being the data matrix, ```2.0``` the BIC penalty parameter and the number of ILS ```restarts``` being set to ```50```.

Similary, in R, one can call:
``` r
flopsearch::flop(X, 2.0, restarts=50)
```

Instead of the number of restarts, it is also possible to set a ```timeout``` in seconds after which the search terminates and returns the best-scoring graph found thus far.
