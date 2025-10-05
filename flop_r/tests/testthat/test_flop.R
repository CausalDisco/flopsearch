test_that("path graph", {
	p <- 10
	W <- matrix(0, nrow = p, ncol = p)
	W[cbind(1:(p-1), 2:p)] <- 1
	X <- matrix(rnorm(10000 * p), nrow = 10000, ncol = p) %*% solve(diag(p) - W)
	X_std <- scale(X)
	G <- flop(X, 2.0, restarts=20)
	expect_true(all(G[cbind(1:(p-1), 2:p)] == 2))
})
