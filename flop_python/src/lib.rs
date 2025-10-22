use ::flop::algo::FlopConfig;
use nalgebra::DMatrix;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

/// Run the FLOP causal discovery algorithm.
///
/// Parameters:
///     data: A data matrix with rows corresponding to observations and columns to variables/nodes.
///     lambda_bic: The penalty parameter of the BIC, a typical value for structure learning is 2.0.
///     restarts: Optional parameter specifying the number of ILS restarts. Either restarts or timeout (below) need to be specified.
///     timeout: Optional parameter specifying a timeout after which the search returns. At least one local search is run up to a local optimum. Either restarts or timeout need to be specified.
///     output_dag: Optional parameter to output a DAG instead of a CPDAG. Default value is False.
///
/// Returns:
///     A matrix encoding a CPDAG or DAG. The entry in row i and column j is 1 in case of a directed edge from i to j and 2 in case of an undirected edge between those nodes (an additional 2 in row j and column i is omitted).
#[pyfunction]
#[pyo3(signature = (data, lambda_bic, *, restarts=None, timeout=None, output_dag=false))]
fn flop<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    lambda_bic: f64,
    restarts: Option<usize>,
    timeout: Option<f64>,
    output_dag: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if restarts.is_none() && timeout.is_none() {
        return Err(PyValueError::new_err(
            "Config error: neither number of restarts nor timeout was specified, e.g., pass restarts=20 as optional argument",
        ));
    }
    let flop_config = FlopConfig::new(lambda_bic, restarts, timeout, false);
    let data_matrix = DMatrix::from(data.as_matrix());
    let g = match ::flop::algo::run(&data_matrix, flop_config) {
        Ok(res) => res,
        Err(err) => Err(PyRuntimeError::new_err(format!("FLOP error: {}", err)))?,
    };

    let mut res = vec![vec![0.0; g.p]; g.p];
    if output_dag {
        for u in 0..g.p {
            for &v in g.parents[u].iter() {
                res[v][u] = 1.0
            }
        }
    } else {
        let g = g.to_cpdag();
        for u in 0..g.p {
            for &v in g.undir_neighbors[u].iter() {
                if u < v {
                    res[u][v] = 2.0;
                }
            }
            for &v in g.out_neighbors[u].iter() {
                res[u][v] = 1.0;
            }
        }
    }

    PyArray2::from_vec2(py, &res).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn flopsearch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::flop, m)?)?;
    Ok(())
}
