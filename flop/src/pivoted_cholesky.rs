use nalgebra::DMatrix;

/// Swaps rows and columns `a` and `b` in the lower triangle of a symmetric matrix stored in column-major packed slice.
/// Assumes a < b and indices within bounds.
fn swap_lower_triangle(packed_lower: &mut [f64], a: usize, b: usize, n: usize) {
    // Region 1: i in 0..a
    // Swap row `a` and `b` for columns before `a`
    (0..a).for_each(|i| packed_lower.swap(a + i * n, b + i * n));

    // Swap diagonal entries
    packed_lower.swap(a * (n + 1), b * (n + 1));

    // Region 2: i in a+1..b
    // Swap column `a` below diagonal with row `b` before diagonal
    ((a + 1)..b).for_each(|i| packed_lower.swap(i + a * n, b + i * n));

    // Region 3: i in b+1..n
    // Swap columns `a` and `b` for rows after `b`
    ((b + 1)..n).for_each(|i| packed_lower.swap(i + a * n, i + b * n));
}

/// Performs pivoted cholesky decomposition on a symmetric positive definite matrix with all-ones diagonal, that is, a full rank **correlation** matrix
/// - Pivots are greedily selected to keep the determinant so far as small as possible
///   - First two pivots correspond to the off-diagonal entry in the correlation matrix with maximal absolute value
///   - Remaining pivots are selected based on miniminal diagonal value
/// - Physical row and column swaps are used (of the lower-triangle only so upper triangle is garbage)
/// - left-looking with diagonals of the trailing submatrix computed ahead to be able to select pivots
pub fn cholesky_left_min_diag(matrix: &DMatrix<f64>) -> Option<(DMatrix<f64>, Vec<usize>)> {
    // No input checks for being square, symmetric, and having an all-1s diagonal;
    // also just panics if matrix is not positive definite
    let n = matrix.nrows();
    let mut ldl = matrix.clone();
    let ldl_slice = ldl.as_mut_slice();
    let mut permutation: Vec<usize> = (0..n).collect();

    // Find the two indices with the largest absolute off-diagonal entry
    // traversing the bottom triangle (as the upper triangle may be garbage)
    struct OffdiagState {
        value: f64,
        rowoffset: usize,
        column: usize,
    }
    let mut offdiag_state = OffdiagState {
        value: f64::NEG_INFINITY,
        rowoffset: 0,
        column: 0,
    };

    ldl_slice
        .chunks_exact(n)
        .enumerate()
        .for_each(|(col_id, col)| {
            col[col_id + 1..n]
                .iter()
                .enumerate()
                .for_each(|(row_id_offset, row)| {
                    let absval = row.abs();
                    if absval > offdiag_state.value {
                        offdiag_state.value = absval;
                        offdiag_state.rowoffset = row_id_offset;
                        offdiag_state.column = col_id;
                    }
                });
        });

    // the order of the first two pivots does not matter
    // (both lead to the same minimal determinants in the first two iterations)
    struct PivotState {
        value: f64,
        id: usize,
    }
    let mut next_pivot = PivotState {
        value: f64::INFINITY,
        id: offdiag_state.column,
    };

    if next_pivot.id == 1 {
        // if the first two pivots are 1 and k > 1,
        // we keep row/column 1 in its position (which will end up being the next pivot after)
        // and instead choose the other row/column k as initial pivot,
        // which instead of two swaps (column 1 and 0 followed by column k and 1)
        // only requires one (column k and 0) for the first two pivots
        next_pivot.id += offdiag_state.rowoffset + 1;
        swap_lower_triangle(ldl_slice, 0, next_pivot.id, n);
        permutation.swap(0, next_pivot.id);
    } else {
        swap_lower_triangle(ldl_slice, 0, next_pivot.id, n);
        permutation.swap(0, next_pivot.id);
        next_pivot.id += offdiag_state.rowoffset + 1;
        swap_lower_triangle(ldl_slice, 1, next_pivot.id, n);
        permutation.swap(1, next_pivot.id);
    }

    // use assumption that matrix is strictly positive definite with all-1s diagonal (i.e. a full rank correlation matrix)
    // so we can skip step=0 iteration below, since the sqrt-diagonal does not change the value, nor division by 1,
    // so all left to do is updating the forward subtractions off the diagonals
    for k in 1..n {
        // forward substract off step_col from only diagonal entries, so we know explained variance and can choose next pivot
        //    k-th diagonal           k-th entry in first column
        ldl_slice[k * n + k] -= ldl_slice[k].powi(2);
    }

    // we only update lower triangle values, so when step > = n-1 there's only a diagonal value left, which we don't update here
    // we start at 1 instead of 0, see above
    // we iterate until < n-1 instead of < n since we forward subtract off the diagonals,
    // so there's nothing to do in the last round other than sqrt-ing the last diagonal entry which we do below
    for step in 1..n - 1 {
        let (left_cols, step_col) = ldl_slice.split_at_mut(step * n);
        let (step_col, right_cols) = step_col.split_at_mut(n);

        let diag_squared = step_col[step];
        if diag_squared <= 0.0 {
            return None;
        }
        let diag = diag_squared.sqrt();
        step_col[step] = diag;

        // --- loop unrolling
        let mut k = 0;
        while k + 3 < step {
            let col_0 = &left_cols[k * n..];
            let col_0 = &col_0[step..n];
            let col_1 = &left_cols[(k + 1) * n..];
            let col_1 = &col_1[step..n];
            let col_2 = &left_cols[(k + 2) * n..];
            let col_2 = &col_2[step..n];
            let col_3 = &left_cols[(k + 3) * n..];
            let col_3 = &col_3[step..n];

            let step_col = &mut step_col[step..n];
            for l in 1..step_col.len() {
                step_col[l] += -col_0[0] * col_0[l]
                    - col_1[0] * col_1[l]
                    - col_2[0] * col_2[l]
                    - col_3[0] * col_3[l];
            }
            k += 4;
        }
        // loop unrolling ---
        for k in k..step {
            let col_l = &left_cols[k * n..];
            let col_l = &col_l[step..n];
            let step_col = &mut step_col[step..n];
            for l in 1..step_col.len() {
                step_col[l] += -col_l[0] * col_l[l];
            }
        }

        next_pivot.value = f64::INFINITY;

        // complete update of step_col
        for k in step + 1..step_col.len() {
            step_col[k] /= diag;

            // and forward substract off step_col from only diagonal entries, so we know explained variance and can choose next pivot
            let diagval = &mut right_cols[(k - step - 1) * n + k];
            *diagval -= step_col[k].powi(2);
            if *diagval < next_pivot.value {
                next_pivot.value = *diagval;
                next_pivot.id = k;
            }
        }
        // swap things around accordingly
        swap_lower_triangle(ldl_slice, step + 1, next_pivot.id, n);
        permutation.swap(step + 1, next_pivot.id);
    }
    // see above, this is the only thing needed when we iterate until < n-1 instead of < n
    ldl_slice[ldl_slice.len() - 1] = ldl_slice[ldl_slice.len() - 1].sqrt();

    Some((ldl, permutation))
}
