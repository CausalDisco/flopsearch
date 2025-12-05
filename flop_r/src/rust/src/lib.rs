#![allow(non_snake_case)]

use extendr_api::prelude::*;
use extendr_api::wrapper::matrix::RMatrix;
use flop::algo::FlopConfig;
use nalgebra::DMatrix;

extendr_module! {
    mod flopsearch;
    fn flop;
}

/// Run the FLOP causal discovery algorithm.
///
/// For the given data and parameters, the FLOP algorithm is run and returns the learned graph as an adjacency matrix.
///
/// @param data A data matrix with rows corresponding to observations and columns to variables/nodes.
/// @param lambdaBic The penalty parameter of the BIC, a typical value for structure learning is 2.0.
/// @param restarts Optional parameter specifying the number of ILS restarts. Either restarts or timeout (below) need to be specified.
/// @param timeout Optional parameter specifying a timeout after which the search returns. At least one local search is run up to a local optimum. Either restarts or timeout need to be specified.
/// @return A matrix encoding a CPDAG. The entry in row i and column j is 1 in case of a directed edge from i to j and 2 in case of an undirected edge between those nodes (in case of an undirected edge, the entry in row j and column i is also 2, that is each undirected edge induce two 2's in the matrix).
/// @examples
/// p <- 10
/// W <- matrix(0, nrow = p, ncol = p)
/// W[cbind(1:(p-1), 2:p)] <- 1
/// X <- matrix(rnorm(10000 * p), nrow = 10000, ncol = p) %*% solve(diag(p) - W)
/// X_std <- scale(X)
/// flop(X, 2.0, restarts=50)
/// @export
#[extendr]
fn flop(
    data: RMatrix<f64>,
    lambdaBic: f64,
    #[default = "NA"] restarts: Option<usize>,
    #[default = "NA"] timeout: Option<f64>,
) -> Result<RMatrix<f64>> {
    if restarts.is_none() && timeout.is_none() {
        return Err(extendr_api::Error::from(
            "Config error: neither number of restarts nor timeout was specified, e.g., pass restarts=50 as optional argument",
        ));
    }
    let flop_config = FlopConfig::new(lambdaBic, restarts, timeout, false);
    let p = data.ncols();
    let n = data.nrows();
    let data_matrix = DMatrix::from_column_slice(n, p, data.data());
    let g = match flop::algo::run(&data_matrix, flop_config) {
        Ok(res) => res,
        Err(err) => Err(extendr_api::Error::from(format!("FLOP error: {}", err)))?,
    };

    let mut res: RMatrix<f64> = RMatrix::new_matrix(p, p, |_, _| 0.0);
    let slice = &mut res.data_mut();

    let g = g.to_cpdag();
    for u in 0..g.p {
        for &v in g.undir_neighbors[u].iter() {
            slice[v * p + u] = 2.0;
        }
        for &v in g.out_neighbors[u].iter() {
            slice[v * p + u] = 1.0;
        }
    }
    Ok(res)
}
