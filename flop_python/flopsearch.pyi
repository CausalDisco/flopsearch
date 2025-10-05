import numpy as np

def flop(
    data: np.ndarray,
    lambda_bic: float,
    *,
    restarts: int,
    timeout: float,
    output_dag: bool = False,
) -> np.ndarray:
    """
    Run the FLOP causal discovery algorithm.

    Parameters
    ----------
    data: A data matrix with rows corresponding to observations and columns to variables/nodes.
    lambda_bic: The penalty parameter of the BIC, a typical value for structure learning is 2.0.
    restarts: Optional parameter specifying the number of ILS restarts. Either restarts or timeout (below) need to be specified.
    timeout: Optional parameter specifying a timeout after which the search returns. At least one local search is run up to a local optimum. Either restarts or timeout need to be specified.
    output_dag: Optional parameter to output a DAG instead of a CPDAG. Default value is False.

    Returns
    -------
    A matrix encoding a CPDAG or DAG. The entry in row i and column j is 1 in case of a directed edge from i to j and 2 in case of an undirected edge between those nodes (an additional 2 in row j and column i is omitted).
    """
    ...
