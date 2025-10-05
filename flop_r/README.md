
<!-- README.md is generated from README.Rmd. Please edit that file -->

# flopsearch

<!-- badges: start -->

<!-- badges: end -->

flopsearch provides an effective and easy-to-use causal discovery
algorithm for linear additive noise models.

## Installation

You can install the development version of flopsearch from
[GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("CausalDisco/flop/flop_r")
```

Note that the name of the installed package is `flopsearch` and can be
loaded with

``` r
library(flopsearch)
```

## Example

A simple example run of the FLOP algorithm provided by flopsearch.

``` r
library(flopsearch)

# set up data
p <- 10
W <- matrix(0, nrow = p, ncol = p)
W[cbind(1:(p-1), 2:p)] <- 1
X <- matrix(rnorm(10000 * p), nrow = 10000, ncol = p) %*% solve(diag(p) - W)
X_std <- scale(X)

# run flop
G <- flop(X, 2.0, restarts=20)
```
