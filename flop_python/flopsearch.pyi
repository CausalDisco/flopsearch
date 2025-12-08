import numpy as np
from typing import Optional

def flop(
    data: np.ndarray,
    lambda_bic: float,
    *,
    restarts: Optional[int] = None,
    timeout: Optional[float] = None,
) -> np.ndarray:
    """
    Run the FLOP causal discovery algorithm.

    Parameters
    ----------
    data: A data matrix with rows corresponding to observations and columns to variables/nodes.
    lambda_bic: The penalty parameter of the BIC, a typical value for structure learning is 2.0.
    restarts: Optional parameter specifying the number of ILS restarts. Either restarts or timeout (below) need to be specified.
    timeout: Optional parameter specifying a timeout after which the search returns. At least one local search is run up to a local optimum. Either restarts or timeout need to be specified.

    Returns
    -------
    A matrix encoding a CPDAG or DAG. The entry in row i and column j is 1 in case of a directed edge from i to j and 2 in case of an undirected edge between those nodes (the entry in row j and column i will also be a 2, that is each undirected edge induces two 2's in the matrix).
    """
    ...
