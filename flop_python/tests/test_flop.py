import flopsearch

import numpy as np
from scipy import linalg


def test_path():
    p = 10
    W = np.diag(np.ones(p - 1), 1)
    X = np.random.randn(10000, p).dot(linalg.inv(np.eye(p) - W))
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    G = flopsearch.flop(X_std, 2.0, restarts=20)
    assert np.all(np.diag(G, k=1) == 2)
